#include "routes.h"

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <regex>
#include <string_view>
#include <thread>
#include <unordered_map>

#include "common/common.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"

namespace fs = std::filesystem;

// ---- SSE event types ----

struct StreamEvent {
    enum class Type { Step, Preview, Result, Error, Done };
    Type type           = Type::Done;
    int step            = 0;
    int total_steps     = 0;
    float time_per_step = 0.f;
    std::vector<std::string> b64_images;
    json parameters;
    std::string error_message;
};

struct StreamState {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<StreamEvent> queue;

    void push(StreamEvent ev) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push_back(std::move(ev));
        }
        cv.notify_one();
    }

    StreamEvent pop() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return !queue.empty(); });
        StreamEvent ev = std::move(queue.front());
        queue.pop_front();
        return ev;
    }
};

// Shared context passed to both generation callbacks via void* data
struct GenCallbackData {
    StreamState* state;
    int total_steps     = 0;
    int preview_interval = 1;
};

// ---- SSE formatting ----

static std::string stream_event_to_sse(const StreamEvent& ev) {
    json j;
    switch (ev.type) {
        case StreamEvent::Type::Step:
            j["type"]        = "step";
            j["step"]        = ev.step;
            j["total_steps"] = ev.total_steps;
            j["time"]        = ev.time_per_step;
            return "data: " + j.dump() + "\n\n";
        case StreamEvent::Type::Preview:
            j["type"]        = "preview";
            j["step"]        = ev.step;
            j["total_steps"] = ev.total_steps;
            j["b64_json"]    = ev.b64_images.empty() ? "" : ev.b64_images[0];
            return "data: " + j.dump() + "\n\n";
        case StreamEvent::Type::Result:
            j["type"]       = "result";
            j["images"]     = ev.b64_images;
            j["parameters"] = ev.parameters;
            return "data: " + j.dump() + "\n\n";
        case StreamEvent::Type::Error:
            j["type"]    = "error";
            j["message"] = ev.error_message;
            return "data: " + j.dump() + "\n\n";
        case StreamEvent::Type::Done:
            return "data: [DONE]\n\n";
    }
    return {};
}

// ---- Image encoding helpers ----

static std::string encode_preview_frame_to_b64(const sd_image_t& img) {
    if (img.data == nullptr || img.width == 0 || img.height == 0) {
        return {};
    }
    // Use JPEG at reduced quality for fast preview delivery
    auto bytes = encode_image_to_vector(EncodedImageFormat::JPEG,
                                        img.data, img.width, img.height, img.channel,
                                        "", 75);
    if (bytes.empty()) {
        return {};
    }
    return base64_encode(bytes);
}

// ---- Generation callbacks ----

static void stream_progress_cb(int step, int steps, float time, void* data) {
    auto* d         = static_cast<GenCallbackData*>(data);
    d->total_steps  = steps;

    StreamEvent ev;
    ev.type          = StreamEvent::Type::Step;
    ev.step          = step;
    ev.total_steps   = steps;
    ev.time_per_step = time;
    d->state->push(std::move(ev));
}

static void stream_preview_cb(int step, int frame_count, sd_image_t* frames, bool /*is_noisy*/, void* data) {
    if (frames == nullptr || frame_count == 0) {
        return;
    }
    auto* d = static_cast<GenCallbackData*>(data);

    // Only emit at the configured interval
    if (d->preview_interval > 0 && step % d->preview_interval != 0) {
        return;
    }

    std::string b64 = encode_preview_frame_to_b64(frames[0]);
    if (b64.empty()) {
        return;
    }

    StreamEvent ev;
    ev.type        = StreamEvent::Type::Preview;
    ev.step        = step;
    ev.total_steps = d->total_steps;
    ev.b64_images.push_back(std::move(b64));
    d->state->push(std::move(ev));
}

// ---- Request parsing (mirrors routes_sdapi.cpp build_sdapi_img_gen_request) ----

static std::string extract_extra_args_stream(std::string& text) {
    std::regex re("<sd_cpp_extra_args>(.*?)</sd_cpp_extra_args>");
    std::smatch match;
    std::string extracted;
    if (std::regex_search(text, match, re)) {
        extracted = match[1].str();
        text      = std::regex_replace(text, re, "");
    }
    return extracted;
}

static std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static enum sample_method_t resolve_sampler_name(std::string name) {
    enum sample_method_t result = str_to_sample_method(name.c_str());
    if (result != SAMPLE_METHOD_COUNT) {
        return result;
    }
    name = to_lower_ascii(name);
    static const std::unordered_map<std::string_view, sample_method_t> aliases{
        {"euler a", EULER_A_SAMPLE_METHOD},
        {"k_euler_a", EULER_A_SAMPLE_METHOD},
        {"euler", EULER_SAMPLE_METHOD},
        {"k_euler", EULER_SAMPLE_METHOD},
        {"heun", HEUN_SAMPLE_METHOD},
        {"k_heun", HEUN_SAMPLE_METHOD},
        {"dpm2", DPM2_SAMPLE_METHOD},
        {"k_dpm_2", DPM2_SAMPLE_METHOD},
        {"lcm", LCM_SAMPLE_METHOD},
        {"ddim", DDIM_TRAILING_SAMPLE_METHOD},
        {"dpm++ 2m", DPMPP2M_SAMPLE_METHOD},
        {"k_dpmpp_2m", DPMPP2M_SAMPLE_METHOD},
        {"res multistep", RES_MULTISTEP_SAMPLE_METHOD},
        {"k_res_multistep", RES_MULTISTEP_SAMPLE_METHOD},
        {"res 2s", RES_2S_SAMPLE_METHOD},
        {"k_res_2s", RES_2S_SAMPLE_METHOD},
    };
    auto it = aliases.find(name);
    return it != aliases.end() ? it->second : SAMPLE_METHOD_COUNT;
}

static void fill_solid_mask(SDImageOwner& mask_owner, int width, int height) {
    const size_t n    = static_cast<size_t>(width) * static_cast<size_t>(height);
    uint8_t* raw_mask = static_cast<uint8_t*>(malloc(n));
    if (raw_mask == nullptr) {
        mask_owner.reset({0, 0, 1, nullptr});
        return;
    }
    std::memset(raw_mask, 255, n);
    mask_owner.reset({(uint32_t)width, (uint32_t)height, 1, raw_mask});
}

static bool parse_stream_img_gen_request(const json& j,
                                         ServerRuntime& runtime,
                                         bool img2img,
                                         ImgGenJobRequest& request,
                                         std::string& error_message) {
    std::string prompt          = j.value("prompt", "");
    std::string negative_prompt = j.value("negative_prompt", "");
    int width                   = j.value("width", 512);
    int height                  = j.value("height", 512);
    int steps                   = j.value("steps", runtime.default_gen_params->sample_params.sample_steps);
    float cfg_scale             = j.value("cfg_scale", runtime.default_gen_params->sample_params.guidance.txt_cfg);
    int64_t seed                = j.value("seed", -1LL);
    int batch_size              = j.value("batch_size", 1);
    int clip_skip               = j.value("clip_skip", -1);
    std::string sampler_name    = j.value("sampler_name", "");
    std::string scheduler_name  = j.value("scheduler", "");

    if (width <= 0 || height <= 0) {
        error_message = "width and height must be positive";
        return false;
    }
    if (prompt.empty()) {
        error_message = "prompt required";
        return false;
    }

    request.gen_params                                = *runtime.default_gen_params;
    request.gen_params.prompt                         = prompt;
    request.gen_params.negative_prompt                = negative_prompt;
    request.gen_params.seed                           = seed;
    request.gen_params.sample_params.sample_steps     = steps;
    request.gen_params.batch_count                    = batch_size;
    request.gen_params.sample_params.guidance.txt_cfg = cfg_scale;
    request.gen_params.width                          = j.value("width", -1);
    request.gen_params.height                         = j.value("height", -1);

    if (!img2img && j.value("enable_hr", false)) {
        request.gen_params.hires_enabled            = true;
        request.gen_params.hires_scale              = j.value("hr_scale", request.gen_params.hires_scale);
        request.gen_params.hires_width              = j.value("hr_resize_x", request.gen_params.hires_width);
        request.gen_params.hires_height             = j.value("hr_resize_y", request.gen_params.hires_height);
        request.gen_params.hires_steps              = j.value("hr_steps", request.gen_params.hires_steps);
        request.gen_params.hires_denoising_strength =
            j.value("denoising_strength", request.gen_params.hires_denoising_strength);
        request.gen_params.hires_upscaler = j.value("hr_upscaler", request.gen_params.hires_upscaler);
    }

    std::string extra_args = extract_extra_args_stream(request.gen_params.prompt);
    if (!extra_args.empty() && !request.gen_params.from_json_str(extra_args)) {
        error_message = "invalid sd_cpp_extra_args";
        return false;
    }

    if (clip_skip > 0) {
        request.gen_params.clip_skip = clip_skip;
    }

    enum sample_method_t sample_method = resolve_sampler_name(sampler_name);
    if (sample_method != SAMPLE_METHOD_COUNT) {
        request.gen_params.sample_params.sample_method = sample_method;
    }

    enum scheduler_t scheduler = str_to_scheduler(scheduler_name.c_str());
    if (scheduler != SCHEDULER_COUNT) {
        request.gen_params.sample_params.scheduler = scheduler;
    }

    if (j.contains("lora") && j["lora"].is_array()) {
        request.gen_params.lora_map.clear();
        request.gen_params.high_noise_lora_map.clear();
        for (const auto& item : j["lora"]) {
            if (!item.is_object()) {
                continue;
            }
            std::string path   = item.value("path", "");
            float multiplier   = item.value("multiplier", 1.0f);
            bool is_high_noise = item.value("is_high_noise", false);
            if (path.empty()) {
                error_message = "lora.path required";
                return false;
            }
            std::string fullpath = get_lora_full_path(runtime, path);
            if (fullpath.empty()) {
                error_message = "invalid lora path: " + path;
                return false;
            }
            if (is_high_noise) {
                request.gen_params.high_noise_lora_map[fullpath] += multiplier;
            } else {
                request.gen_params.lora_map[fullpath] += multiplier;
            }
        }
    }

    if (img2img) {
        const int expected_w = request.gen_params.width_and_height_are_set() ? request.gen_params.width : 0;
        const int expected_h = request.gen_params.width_and_height_are_set() ? request.gen_params.height : 0;

        if (j.contains("init_images") && j["init_images"].is_array() && !j["init_images"].empty()) {
            if (decode_base64_image(j["init_images"][0].get<std::string>(), 3,
                                    expected_w, expected_h, request.gen_params.init_image)) {
                const sd_image_t& img = request.gen_params.init_image.get();
                request.gen_params.set_width_and_height_if_unset(img.width, img.height);
            }
        }

        if (j.contains("mask") && j["mask"].is_string()) {
            if (decode_base64_image(j["mask"].get<std::string>(), 1,
                                    expected_w, expected_h, request.gen_params.mask_image)) {
                const sd_image_t& img = request.gen_params.mask_image.get();
                request.gen_params.set_width_and_height_if_unset(img.width, img.height);
            }
            sd_image_t& mask            = request.gen_params.mask_image.get();
            bool inpainting_mask_invert = j.value("inpainting_mask_invert", 0) != 0;
            if (inpainting_mask_invert && mask.data != nullptr) {
                for (uint32_t i = 0; i < mask.width * mask.height; ++i) {
                    mask.data[i] = 255 - mask.data[i];
                }
            }
        } else {
            fill_solid_mask(request.gen_params.mask_image,
                            request.gen_params.get_resolved_width(),
                            request.gen_params.get_resolved_height());
        }

        float denoising_strength = j.value("denoising_strength", -1.f);
        if (denoising_strength >= 0.f) {
            request.gen_params.strength = std::min(denoising_strength, 1.0f);
        }
    }

    if (j.contains("extra_images") && j["extra_images"].is_array()) {
        for (const auto& extra : j["extra_images"]) {
            if (!extra.is_string()) {
                continue;
            }
            SDImageOwner img_owner;
            const int ew = request.gen_params.width_and_height_are_set() ? request.gen_params.width : 0;
            const int eh = request.gen_params.width_and_height_are_set() ? request.gen_params.height : 0;
            if (decode_base64_image(extra.get<std::string>(), 3, ew, eh, img_owner)) {
                const sd_image_t& img = img_owner.get();
                request.gen_params.set_width_and_height_if_unset(img.width, img.height);
                request.gen_params.ref_images.push_back(std::move(img_owner));
            }
        }
    }

    if (!request.gen_params.resolve_and_validate(IMG_GEN, "", runtime.ctx_params->hires_upscalers_dir, true)) {
        error_message = "invalid params";
        return false;
    }
    return true;
}

// ---- Endpoint registration ----

void register_stream_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;

    auto stream_any2img = [runtime](const httplib::Request& req, httplib::Response& res, bool img2img) {
        try {
            if (req.body.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }
            if (!runtime_supports_generation_mode(*runtime, IMG_GEN)) {
                res.status = 400;
                res.set_content(
                    json({{"error", unsupported_generation_mode_error(IMG_GEN)}}).dump(),
                    "application/json");
                return;
            }

            json j = json::parse(req.body);
            ImgGenJobRequest request;
            std::string error_message;
            if (!parse_stream_img_gen_request(j, *runtime, img2img, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }

            // Streaming-specific options
            int preview_interval         = j.value("stream_preview_interval", 1);
            std::string preview_mode_str = j.value("stream_preview_mode", "proj");
            enum preview_t preview_mode  = str_to_preview(preview_mode_str.c_str());
            if (preview_mode == PREVIEW_COUNT) {
                preview_mode = PREVIEW_PROJ;
            }

            res.set_header("Cache-Control", "no-cache");
            res.set_header("X-Accel-Buffering", "no");

            auto state = std::make_shared<StreamState>();

            // Generation runs in a background thread so the content provider
            // can stream events as they arrive without blocking httplib.
            std::thread gen_thread(
                [runtime, request = std::move(request), state,
                 preview_interval, preview_mode, j]() mutable {

                    GenCallbackData cb_data;
                    cb_data.state            = state.get();
                    cb_data.preview_interval = preview_interval;

                    std::lock_guard<std::mutex> lock(*runtime->sd_ctx_mutex);

                    sd_set_progress_callback(stream_progress_cb, &cb_data);

                    if (preview_interval > 0) {
                        sd_set_preview_callback(stream_preview_cb,
                                                preview_mode, 1,
                                                /*denoised=*/true, /*noisy=*/false,
                                                &cb_data);
                    }

                    int num_results            = request.gen_params.batch_count;
                    sd_img_gen_params_t params = request.to_sd_img_gen_params_t();
                    sd_image_t* raw_results    = generate_image(runtime->sd_ctx, &params);
                    SDImageVec results;
                    results.adopt(raw_results, num_results);

                    // Clear global callbacks immediately after generation completes
                    sd_set_progress_callback(nullptr, nullptr);
                    sd_set_preview_callback(nullptr, PREVIEW_NONE, 0, false, false, nullptr);

                    if (results.empty()) {
                        StreamEvent ev;
                        ev.type          = StreamEvent::Type::Error;
                        ev.error_message = "generate_image returned no results";
                        state->push(std::move(ev));
                    } else {
                        StreamEvent ev;
                        ev.type       = StreamEvent::Type::Result;
                        ev.parameters = j;

                        for (int i = 0; i < num_results; ++i) {
                            if (results[i].data == nullptr) {
                                continue;
                            }
                            std::string meta = request.gen_params.embed_image_metadata
                                ? get_image_params(*runtime->ctx_params,
                                                   request.gen_params,
                                                   request.gen_params.seed + i)
                                : "";
                            auto image_bytes = encode_image_to_vector(
                                EncodedImageFormat::PNG,
                                results[i].data,
                                results[i].width,
                                results[i].height,
                                results[i].channel,
                                meta);
                            if (!image_bytes.empty()) {
                                ev.b64_images.push_back(base64_encode(image_bytes));
                            }
                        }
                        state->push(std::move(ev));
                    }

                    StreamEvent done_ev;
                    done_ev.type = StreamEvent::Type::Done;
                    state->push(std::move(done_ev));
                });
            gen_thread.detach();

            // Content provider: pull events from the queue and write SSE lines
            res.set_chunked_content_provider(
                "text/event-stream",
                [state](size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {
                    StreamEvent ev      = state->pop();
                    std::string sse_str = stream_event_to_sse(ev);
                    bool ok             = sink.write(sse_str.c_str(), sse_str.size());

                    if (ev.type == StreamEvent::Type::Done ||
                        ev.type == StreamEvent::Type::Error) {
                        sink.done();
                        return false;
                    }
                    return ok;
                });

        } catch (const std::exception& e) {
            res.status = 500;
            json err;
            err["error"]   = "server_error";
            err["message"] = e.what();
            res.set_content(err.dump(), "application/json");
        }
    };

    svr.Post("/sdapi/v1/txt2img/stream",
             [stream_any2img](const httplib::Request& req, httplib::Response& res) {
                 stream_any2img(req, res, false);
             });

    svr.Post("/sdapi/v1/img2img/stream",
             [stream_any2img](const httplib::Request& req, httplib::Response& res) {
                 stream_any2img(req, res, true);
             });
}
