#pragma once

#include <string>

#include "httplib.h"

// Configuration for the llama-server proxy.
struct LLMProxyConfig {
    std::string host;
    int port = 8081;

    bool enabled() const { return !host.empty(); }
};

// Parse "http://host:port" (or "host:port") into an LLMProxyConfig.
// Returns a config with empty host if the URL is empty or malformed.
inline LLMProxyConfig parse_llm_proxy_url(const std::string& url) {
    LLMProxyConfig cfg;
    if (url.empty()) {
        return cfg;
    }

    std::string stripped = url;

    // Strip scheme
    for (const auto* scheme : {"https://", "http://"}) {
        const std::string s(scheme);
        if (stripped.substr(0, s.size()) == s) {
            stripped = stripped.substr(s.size());
            break;
        }
    }

    // Strip trailing slash
    if (!stripped.empty() && stripped.back() == '/') {
        stripped.pop_back();
    }

    auto colon = stripped.rfind(':');
    if (colon != std::string::npos) {
        cfg.host = stripped.substr(0, colon);
        try {
            cfg.port = std::stoi(stripped.substr(colon + 1));
        } catch (...) {
            cfg.host.clear();
        }
    } else {
        cfg.host = stripped;
    }
    return cfg;
}

// Launch llama-server as a child process.
// Returns the child PID on success, -1 on failure.
// POSIX-only; on Windows this is a no-op returning -1.
int launch_llm_subprocess(const std::string& binary,
                          const std::string& model,
                          int port,
                          int n_threads = -1);

// Poll GET /health on the proxy until it responds 200 or timeout_ms elapses.
// Returns true if the server became ready in time.
bool wait_for_llm_ready(const LLMProxyConfig& cfg, int timeout_ms = 15000);

// Send SIGTERM (POSIX) or TerminateProcess (Windows) to a launched subprocess.
void kill_llm_subprocess(int pid);
