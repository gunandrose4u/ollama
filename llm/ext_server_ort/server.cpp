// MIT License

// Copyright (c) 2023 Georgi Gerganov

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// #include "common.h"
// #include "llama.h"
// #include "grammar-parser.h"
// #include "utils.hpp"

// #include "../llava/clip.h"
// #include "../llava/llava.h"

// #include "stb_image.h"
#include <iostream>
#include <span>
#include "ort_genai.h"

#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif
// increase max payload length to allow use of larger context size
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576
#include "httplib.h"
#include "json.hpp"
// #include "utils.hpp"


#include <cstddef>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <atomic>
#include <signal.h>

using json = nlohmann::json;

struct server_params {
    std::string hostname = "127.0.0.1";
    std::vector<std::string> api_keys;
    std::string public_path = "examples/server/public";
    std::string chat_template = "";
    std::string model = "";
    int32_t port = 8080;
    int32_t read_timeout = 600;
    int32_t write_timeout = 600;
    int32_t n_ctx = 2048;
    bool slots_endpoint = true;
    bool metrics_endpoint = false;
    int n_threads_http = -1;
};

bool server_verbose = false;
bool server_log_json = false;

static json format_tokenizer_response(const std::vector<int32_t> &tokens)
{
    json t;
    t["tokens"] = tokens;
    return t;
}

static json format_streaming_chat_response(const char* content, bool stop)
{
    json t = {
        {"content", content},
        {"multimodal", false},
        {"slot_id", false},
        {"stop", false}
    };

    return t;
}

static void server_params_parse(int argc, char **argv, server_params &sparams)
{
    server_params default_sparams;
    std::string arg;
    bool invalid_param = false;

    for (int i = 1; i < argc; i++)
    {
        arg = argv[i];
        if (arg == "--port")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.port = std::stoi(argv[i]);
        }
        else if (arg == "--host")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.hostname = argv[i];
        }
        else if (arg == "--path")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.public_path = argv[i];
        }
        else if (arg == "--api-key")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.api_keys.emplace_back(argv[i]);
        }
        else if (arg == "--api-key-file")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            std::ifstream key_file(argv[i]);
            if (!key_file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
                invalid_param = true;
                break;
            }
            std::string key;
            while (std::getline(key_file, key)) {
               if (key.size() > 0) {
                   sparams.api_keys.push_back(key);
               }
            }
            key_file.close();
        }
        else if (arg == "-m" || arg == "--model")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.model = argv[i];
        }
        else if (arg == "--timeout" || arg == "-to")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.read_timeout = std::stoi(argv[i]);
            sparams.write_timeout = std::stoi(argv[i]);
        }
        else if (arg == "-c" || arg == "--ctx-size" || arg == "--ctx_size")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.n_ctx = std::stoi(argv[i]);
        }
        else if (arg == "--threads-http")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.n_threads_http = std::stoi(argv[i]);
        }
        else if (arg == "--log-format")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            if (std::strcmp(argv[i], "json") == 0)
            {
                server_log_json = true;
            }
            else if (std::strcmp(argv[i], "text") == 0)
            {
                server_log_json = false;
            }
            else
            {
                invalid_param = true;
                break;
            }
        }
        else if (arg == "--slots-endpoint-disable")
        {
            sparams.slots_endpoint = false;
        }
        else if (arg == "--metrics")
        {
            sparams.metrics_endpoint = true;
        }
        else if (arg == "--chat-template")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.chat_template = argv[i];
        }
    }
}

template <typename T>
static T json_value(const json &body, const std::string &key, const T &default_value) {
    // Fallback null to default value
    return body.contains(key) && !body.at(key).is_null()
        ? body.value(key, default_value)
        : default_value;
}

int main(int argc, char **argv) {
    // own arguments required by this example
    server_params sparams;
    server_params_parse(argc, argv, sparams);

    // load the model
    std::cout << "Loading model ..." << std::endl;
    std::cout << sparams.model << std::endl;
    auto ort_model = OgaModel::Create(sparams.model.c_str());
    std::cout << "Creating tokenizer..." << std::endl;
    auto tokenizer = OgaTokenizer::Create(*ort_model);
    auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

    httplib::Server svr;

    svr.set_default_headers({{"Server", "ort-genai"}});

    // CORS preflight
    svr.Options(R"(.*)", [](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "POST");
        res.set_header("Access-Control-Allow-Headers", "*");
    });

    svr.Get("/health", [&](const httplib::Request& req, httplib::Response& res) {
        res.status = 200;
        json health = {
                        {"status",           "ok"},
                        {"slots_idle",       1},
                        {"slots_processing", 0}};
        res.set_content(health.dump(), "application/json");
    });

    if (sparams.slots_endpoint) {
        svr.Get("/slots", [&](const httplib::Request&, httplib::Response& res) {
            res.status = 200; // HTTP OK
        });
    }

    if (sparams.metrics_endpoint) {
        svr.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
            res.status = 200; // HTTP OK
        });
    }

    svr.set_exception_handler([](const httplib::Request &, httplib::Response &res, std::exception_ptr ep)
            {
                const char fmt[] = "500 Internal Server Error\n%s";
                char buf[BUFSIZ];
                try
                {
                    std::rethrow_exception(std::move(ep));
                }
                catch (std::exception &e)
                {
                    snprintf(buf, sizeof(buf), fmt, e.what());
                }
                catch (...)
                {
                    snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
                }
                res.set_content(buf, "text/plain; charset=utf-8");
                res.status = 500;
            });

    svr.set_error_handler([](const httplib::Request &, httplib::Response &res)
            {
                if (res.status == 401)
                {
                    res.set_content("Unauthorized", "text/plain; charset=utf-8");
                }
                if (res.status == 400)
                {
                    res.set_content("Invalid request", "text/plain; charset=utf-8");
                }
                else if (res.status == 404)
                {
                    res.set_content("File Not Found", "text/plain; charset=utf-8");
                    res.status = 404;
                }
            });

    // set timeouts and change hostname and port
    svr.set_read_timeout (sparams.read_timeout);
    svr.set_write_timeout(sparams.write_timeout);

    if (!svr.bind_to_port(sparams.hostname, sparams.port))
    {
        fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d\n\n", sparams.hostname.c_str(), sparams.port);
        return 1;
    }

    // Set the base directory for serving static files
    svr.set_base_dir(sparams.public_path);

    std::unordered_map<std::string, std::string> log_data;
    log_data["hostname"] = sparams.hostname;
    log_data["port"] = std::to_string(sparams.port);

    if (sparams.api_keys.size() == 1) {
        log_data["api_key"] = "api_key: ****" + sparams.api_keys[0].substr(sparams.api_keys[0].length() - 4);
    } else if (sparams.api_keys.size() > 1) {
        log_data["api_key"] = "api_key: " + std::to_string(sparams.api_keys.size()) + " keys loaded";
    }

    if (sparams.n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        sparams.n_threads_http = std::max(2 + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    log_data["n_threads_http"] =  std::to_string(sparams.n_threads_http);
    svr.new_task_queue = [&sparams] { return new httplib::ThreadPool(sparams.n_threads_http); };

    printf("HTTP server listening %s\n", log_data["hostname"].c_str());
    // run the HTTP server in a thread - see comment below
    std::thread t([&]()
            {
                if (!svr.listen_after_bind())
                {
                    return 1;
                }

                return 0;
            });

    // this is only called if no index.html is found in the public --path
    svr.Get("/", [](const httplib::Request &, httplib::Response &res)
            {
                res.set_content("server running", "text/plain; charset=utf-8");
                res.status = 200; // Unauthorized
                return true;
            });

    svr.Post("/tokenize", [&tokenizer](const httplib::Request &req, httplib::Response &res)
            {
                res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
                const json body = json::parse(req.body);
                std::vector<int32_t> tokens;
                auto sequences = OgaSequences::Create();
                if (body.count("content") != 0)
                {
                    tokenizer->Encode(body["content"].dump().c_str(), *sequences);
                    auto token_seq = sequences->SequenceData(0);
                    auto size = sequences->SequenceCount(0);
                    for (int i = 0; i < size; i++)
                    {
                        auto p = token_seq + i;
                        tokens.push_back(*p);
                        printf("%d ", *p);
                    }
                }
                const json data = format_tokenizer_response(tokens);
                return res.set_content(data.dump(), "application/json; charset=utf-8");
            });

    svr.Post("/completion", [&tokenizer, &tokenizer_stream, &ort_model](const httplib::Request &req, httplib::Response &res)
            {
                res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
                json data = json::parse(req.body);
                std::string prompt_str = data["prompt"].dump();
                printf("data: %s\n", data.dump().c_str());
                std::cout << prompt_str << std::flush;

                if (!json_value(data, "stream", false)) {
                    res.status = 404;
                    res.set_content("Non-streaming is not supported.", "text/plain; charset=utf-8");
                } else {
                    printf("streaming\n");
                    const auto chunked_content_provider = [prompt_str, &tokenizer, &tokenizer_stream, &ort_model](size_t, httplib::DataSink & sink)
                    {
                        printf("chunked_content_provider\n");
                        auto sequences = OgaSequences::Create();

                        std::string toSearch = "<|user|>";
                        auto pos = prompt_str.rfind(toSearch);

                        if (pos != std::string::npos) {
                            auto prompt_str_new = prompt_str.substr(pos);
                            printf("Encode %s\n", prompt_str_new.c_str());
                            tokenizer->Encode(prompt_str_new.c_str(), *sequences);
                            printf("Encode done\n");
                        } else {
                            std::cout << "<|user|> not found.\n";
                            printf("Encode %s\n", prompt_str.c_str());
                            tokenizer->Encode(prompt_str.c_str(), *sequences);
                            printf("Encode done\n");
                        }

                        auto params = OgaGeneratorParams::Create(*ort_model);
                        params->SetSearchOption("max_length", 4096);
                        params->SetInputSequences(*sequences);
                        auto generator = OgaGenerator::Create(*ort_model, *params);

                        while (!generator->IsDone()) {
                            generator->ComputeLogits();
                            generator->GenerateNextToken();

                            const auto num_tokens = generator->GetSequenceCount(0);
                            const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
                            const auto decode_c_str = tokenizer_stream->Decode(new_token);
                            // std::cout << decode_c_str << std::flush;

                            json json_res = format_streaming_chat_response(decode_c_str, false);
                            const std::string str =
                                    "data: " +
                                    json_res.dump(-1, ' ', false, json::error_handler_t::replace) +
                                    "\n\n";
                            // printf("to_send: \n%s\n", str.c_str());
                            if (!sink.write(str.c_str(), str.size()))
                            {
                                return false;
                            }
                        }

                        json json_res = format_streaming_chat_response("", true);
                        const std::string str =
                            "data: " +
                            json_res.dump(-1, ' ', false, json::error_handler_t::replace) +
                            "\n\n";
                        if (!sink.write(str.c_str(), str.size()))
                        {
                            return false;
                        }
                        sink.done();
                        return true;
                    };

                    auto on_complete = [] (bool)
                    {
                        // cancel
                        printf("on_complete finished\n");
                    };

                    res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
                }
            });
    // svr.stop();
    t.join();
    return 0;
    // while (true)
    // {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(500));
}
