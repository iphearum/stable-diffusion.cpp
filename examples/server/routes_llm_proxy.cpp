#include "routes.h"
#include "llm_proxy.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#ifndef _WIN32
#    include <csignal>
#    include <sys/types.h>
#    include <sys/wait.h>
#    include <unistd.h>
#endif

#include "common/common.h"

// ---- Subprocess management ----

int launch_llm_subprocess(const std::string& binary,
                          const std::string& model,
                          int port,
                          int n_threads) {
#ifdef _WIN32
    (void)binary;
    (void)model;
    (void)port;
    (void)n_threads;
    LOG_WARN("auto-launch of llama-server is not supported on Windows; "
             "start it manually and use --llm-proxy\n");
    return -1;
#else
    pid_t pid = fork();
    if (pid < 0) {
        LOG_ERROR("fork() failed: %s\n", strerror(errno));
        return -1;
    }
    if (pid == 0) {
        // Child: exec llama-server
        std::string port_str    = std::to_string(port);
        std::string threads_str = std::to_string(n_threads > 0 ? n_threads : 4);

        const char* args[] = {
            binary.c_str(),
            "--model", model.c_str(),
            "--port", port_str.c_str(),
            "--threads", threads_str.c_str(),
            "--host", "127.0.0.1",
            "--log-disable",
            nullptr,
        };
        execvp(binary.c_str(), const_cast<char* const*>(args));
        // execvp only returns on error
        std::fprintf(stderr, "execvp(%s) failed: %s\n", binary.c_str(), strerror(errno));
        _exit(1);
    }
    LOG_INFO("launched llama-server (pid %d) on port %d\n", (int)pid, port);
    return static_cast<int>(pid);
#endif
}

bool wait_for_llm_ready(const LLMProxyConfig& cfg, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);

    while (std::chrono::steady_clock::now() < deadline) {
        httplib::Client cli(cfg.host, cfg.port);
        cli.set_connection_timeout(1);
        cli.set_read_timeout(1);
        auto res = cli.Get("/health");
        if (res && res->status == 200) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return false;
}

void kill_llm_subprocess(int pid) {
    if (pid <= 0) {
        return;
    }
#ifdef _WIN32
    (void)pid;
#else
    ::kill(static_cast<pid_t>(pid), SIGTERM);
    int status = 0;
    ::waitpid(static_cast<pid_t>(pid), &status, 0);
    LOG_INFO("llama-server (pid %d) stopped\n", pid);
#endif
}

// ---- Proxy helpers ----

// Forward headers from the incoming request to the upstream call,
// excluding hop-by-hop headers that must not be forwarded.
static httplib::Headers filter_proxy_headers(const httplib::Headers& in) {
    static const std::set<std::string> blocked{
        "host", "connection", "transfer-encoding",
        "content-length", "accept-encoding",
    };
    httplib::Headers out;
    for (const auto& kv : in) {
        std::string lower;
        for (char c : kv.first) {
            lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (blocked.find(lower) == blocked.end()) {
            out.emplace(kv.first, kv.second);
        }
    }
    return out;
}

// Detect whether the request body contains "stream":true to decide
// whether to use chunked streaming on the response.
static bool request_wants_stream(const std::string& body) {
    // Quick scan: look for "stream" followed by true without full JSON parse.
    // Handles the common case; pathological formatting is ignored.
    auto pos = body.find("\"stream\"");
    if (pos == std::string::npos) {
        return false;
    }
    auto after = body.find_first_not_of(" \t\r\n:", pos + 8);
    if (after == std::string::npos) {
        return false;
    }
    return body.substr(after, 4) == "true";
}

// Proxy a single request to llama-server, handling both streaming and
// non-streaming responses.
static void proxy_to_llm(const httplib::Request& req,
                          httplib::Response& res,
                          const LLMProxyConfig& cfg,
                          const std::string& target_path) {
    httplib::Headers fwd_headers = filter_proxy_headers(req.headers);
    bool is_stream               = request_wants_stream(req.body);

    if (is_stream) {
        // Stream SSE chunks directly through to the client.
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");

        // Capture everything we need by value so the lambda outlives this frame.
        std::string body        = req.body;
        std::string host        = cfg.host;
        int port                = cfg.port;
        std::string path        = target_path;
        httplib::Headers hdrs   = std::move(fwd_headers);

        res.set_chunked_content_provider(
            "text/event-stream",
            [host, port, path, body, hdrs](size_t /*offset*/,
                                           httplib::DataSink& sink) mutable -> bool {
                httplib::Client cli(host, port);
                cli.set_read_timeout(300);
                cli.set_write_timeout(30);

                cli.Post(path, hdrs, body, "application/json",
                         [&sink](const char* data, size_t len) -> bool {
                             return sink.write(data, len);
                         });
                sink.done();
                return false;
            });
    } else {
        httplib::Client cli(cfg.host, cfg.port);
        cli.set_read_timeout(300);
        cli.set_write_timeout(30);

        auto llm_res = cli.Post(target_path, fwd_headers, req.body, "application/json");
        if (llm_res) {
            res.status = llm_res->status;
            std::string ct = llm_res->get_header_value("Content-Type");
            if (ct.empty()) {
                ct = "application/json";
            }
            res.set_content(llm_res->body, ct);
        } else {
            res.status = 502;
            res.set_content(R"({"error":{"message":"llm proxy unreachable","type":"proxy_error"}})",
                            "application/json");
        }
    }
}

// ---- Endpoint registration ----

void register_llm_proxy_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    if (!rt.llm_proxy || !rt.llm_proxy->enabled()) {
        return;
    }

    LLMProxyConfig* cfg    = rt.llm_proxy;
    ServerRuntime* runtime = &rt;

    // --- Merged /v1/models ---
    // Override the existing GET /v1/models registered by routes_openai.cpp.
    // httplib uses the last-registered handler for a path, so registering
    // here (after routes_openai.cpp) replaces it.
    svr.Get("/v1/models", [runtime, cfg](const httplib::Request&, httplib::Response& res) {
        using json = nlohmann::json;

        // SD model entry (same as routes_openai.cpp)
        json sd_entry = {{"id", "sd-cpp-local"}, {"object", "model"}, {"owned_by", "local"}};

        json merged_data = json::array();
        merged_data.push_back(sd_entry);

        // Fetch LLM models from llama-server
        httplib::Client cli(cfg->host, cfg->port);
        cli.set_connection_timeout(2);
        cli.set_read_timeout(5);
        auto llm_res = cli.Get("/v1/models");
        if (llm_res && llm_res->status == 200) {
            try {
                auto llm_json = json::parse(llm_res->body);
                if (llm_json.contains("data") && llm_json["data"].is_array()) {
                    for (const auto& m : llm_json["data"]) {
                        merged_data.push_back(m);
                    }
                }
            } catch (...) {
            }
        }

        json r;
        r["object"] = "list";
        r["data"]   = merged_data;
        res.set_content(r.dump(), "application/json");
    });

    // --- Health check ---
    svr.Get("/health", [cfg](const httplib::Request&, httplib::Response& res) {
        httplib::Client cli(cfg->host, cfg->port);
        cli.set_connection_timeout(1);
        cli.set_read_timeout(2);
        auto llm_res = cli.Get("/health");

        using json = nlohmann::json;
        json r;
        r["sd_status"]  = "ok";
        r["llm_status"] = (llm_res && llm_res->status == 200) ? "ok" : "unavailable";
        r["status"]     = "ok";
        res.set_content(r.dump(), "application/json");
    });

    // --- LLM passthrough routes ---

    // Chat completions (streaming-aware)
    svr.Post("/v1/chat/completions",
             [cfg](const httplib::Request& req, httplib::Response& res) {
                 proxy_to_llm(req, res, *cfg, "/v1/chat/completions");
             });

    // Legacy completions (streaming-aware)
    svr.Post("/v1/completions",
             [cfg](const httplib::Request& req, httplib::Response& res) {
                 proxy_to_llm(req, res, *cfg, "/v1/completions");
             });

    // Embeddings
    svr.Post("/v1/embeddings",
             [cfg](const httplib::Request& req, httplib::Response& res) {
                 proxy_to_llm(req, res, *cfg, "/v1/embeddings");
             });

    // llama.cpp native completion endpoint
    svr.Post("/completion",
             [cfg](const httplib::Request& req, httplib::Response& res) {
                 proxy_to_llm(req, res, *cfg, "/completion");
             });

    // Tokenize / detokenize
    svr.Post("/tokenize",
             [cfg](const httplib::Request& req, httplib::Response& res) {
                 proxy_to_llm(req, res, *cfg, "/tokenize");
             });

    svr.Post("/detokenize",
             [cfg](const httplib::Request& req, httplib::Response& res) {
                 proxy_to_llm(req, res, *cfg, "/detokenize");
             });

    // Props / slots (GET, no body)
    svr.Get("/props",
            [cfg](const httplib::Request&, httplib::Response& res) {
                httplib::Client cli(cfg->host, cfg->port);
                cli.set_read_timeout(5);
                auto r = cli.Get("/props");
                if (r) {
                    res.status = r->status;
                    res.set_content(r->body,
                                    r->get_header_value("Content-Type", "application/json"));
                } else {
                    res.status = 502;
                    res.set_content(R"({"error":"llm proxy unreachable"})", "application/json");
                }
            });

    svr.Get("/slots",
            [cfg](const httplib::Request&, httplib::Response& res) {
                httplib::Client cli(cfg->host, cfg->port);
                cli.set_read_timeout(5);
                auto r = cli.Get("/slots");
                if (r) {
                    res.status = r->status;
                    res.set_content(r->body,
                                    r->get_header_value("Content-Type", "application/json"));
                } else {
                    res.status = 502;
                    res.set_content(R"({"error":"llm proxy unreachable"})", "application/json");
                }
            });

    LOG_INFO("LLM proxy enabled → %s:%d\n", cfg->host.c_str(), cfg->port);
}
