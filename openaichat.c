#define _GNU_SOURCE
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <curl/curl.h>
#include <json-c/json.h>

#include "config.h"

/* *** MODIFIED FOR OPEN WEBUI *** */
#define DEFAULT_WEBUI_BASE "http://192.168.50.5:8080"
#define DEFAULT_OLLAMA_URL DEFAULT_WEBUI_BASE "/api/chat/completions"
#define SYSTEM_PROMPT                                                                                 \
    "You are a helpful and creative AI assistant in a conversation with other friendly AI "        \
    "companions. The user has started the conversation with a topic. Engage in a natural, "         \
    "back-and-forth discussion, building on what the other AI says. Keep your responses "           \
    "concise. Speak directly as your assigned participant without narrating the conversation "      \
    "structure, and never reveal your internal thinking—share only your final reply.\n\n"

#define MAX_PARTICIPANTS 6
#define MAX_NAME_LENGTH 64
#define MAX_MODEL_LENGTH 256
#define MIN_TURNS 1
#define MAX_TURNS 12
#define DEFAULT_PORT 4000
#define FALLBACK_PORT_STEPS 3
#define READ_BUFFER_CHUNK 4096

struct MemoryStruct {
    char *memory;
    size_t size;
};

struct Participant {
    char name[MAX_NAME_LENGTH];
    char model[MAX_MODEL_LENGTH];
    char display_model[MAX_MODEL_LENGTH];
};

typedef int (*message_callback)(json_object *message, void *user_data);

static void send_http_response(int client_fd, const char *status, const char *content_type, const char *body);
static void send_http_error(int client_fd, const char *status, const char *message);
static void handle_diagnostics_request(int client_fd, const char *ollama_url);

static const char *get_ollama_url(void) {
    const char *env = getenv("OLLAMA_URL");
    if (env && *env) {
        return env;
    }
    return DEFAULT_OLLAMA_URL;
}

/* *** ADDED FOR OPEN WEBUI *** */
static int has_visible_text(const char *value) {
    if (!value) {
        return 0;
    }

    while (*value) {
        if (!isspace((unsigned char)*value)) {
            return 1;
        }
        value++;
    }

    return 0;
}

static const char *get_webui_key(void) {
#ifdef WEBUI_API_KEY
    if (has_visible_text(WEBUI_API_KEY) &&
        strcmp(WEBUI_API_KEY, "your-real-api-key-goes-here") != 0) {
        return WEBUI_API_KEY;
    }
#endif

    const char *env_key = getenv("WEBUI_API_KEY");

    if (has_visible_text(env_key)) {
        return env_key;
    }

    return NULL;
}

static void trim_leading_whitespace(char *text);
static void trim_trailing_whitespace(char *text);
static char *append_to_history(char *history, const char *text);
static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp);







static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp);







static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;
    char *ptr = realloc(mem->memory, mem->size + realsize + 1);
    if (!ptr) {
        fprintf(stderr, "Error: not enough memory (realloc returned NULL)\n");
        return 0;
    }
    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = '\0';
    return realsize;
}

static void trim_leading_whitespace(char *text) {
    char *start = NULL;

    if (!text) {
        return;
    }

    start = text;
    while (*start && isspace((unsigned char)*start)) {
        start++;
    }
    if (start != text) {
        memmove(text, start, strlen(start) + 1);
    }
}

static void trim_trailing_whitespace(char *text) {
    size_t length = 0;

    if (!text) {
        return;
    }

    length = strlen(text);
    while (length > 0) {
        unsigned char ch = (unsigned char)text[length - 1];
        if (!isspace(ch)) {
            break;
        }
        text[length - 1] = '\0';
        length--;
    }
}

static void remove_tagged_section(char *text, const char *open_tag, const char *close_tag) {
    size_t open_len = 0;
    size_t close_len = 0;

    if (!text || !open_tag || !close_tag) {
        return;
    }

    open_len = strlen(open_tag);
    close_len = strlen(close_tag);
    if (open_len == 0 || close_len == 0) {
        return;
    }

    while (*text) {
        char *start = strcasestr(text, open_tag);
        char *end = NULL;

        if (!start) {
            return;
        }

        end = strcasestr(start + open_len, close_tag);
        if (end) {
            end += close_len;
            memmove(start, end, strlen(end) + 1);
        } else {
            *start = '\0';
            return;
        }
    }
}

static void remove_leading_metadata_block(char *text) {
    static const char *const prefixes[] = {"thought:",         "thinking:",      "thoughts:",
                                           "analysis:",        "reasoning:",     "chain of thought:",
                                           "internal monologue:", "scratchpad:", "plan:"};
    static const char *const markers[] = {"\nanswer:",      "\nfinal answer:", "\nresponse:",
                                          "\nreply:",       "\nfinal:",        "\noutput:",
                                          "\nresult:"};
    char *start = NULL;

    if (!text) {
        return;
    }

    trim_leading_whitespace(text);
    start = text;

    for (size_t i = 0; i < (sizeof(prefixes) / sizeof(prefixes[0])); ++i) {
        size_t prefix_len = strlen(prefixes[i]);
        if (strncasecmp(start, prefixes[i], prefix_len) == 0) {
            char *search_start = start + prefix_len;
            char *removal_end = NULL;

            for (size_t j = 0; j < (sizeof(markers) / sizeof(markers[0])); ++j) {
                char *candidate = strcasestr(search_start, markers[j]);
                if (candidate && (!removal_end || candidate < removal_end)) {
                    removal_end = candidate + 1; /* retain the newline for trimming */
                }
            }

            char *double_newline = strstr(search_start, "\n\n");
            if (double_newline && (!removal_end || double_newline < removal_end)) {
                removal_end = double_newline + 2;
            }

            char *crlf_double = strstr(search_start, "\r\n\r\n");
            if (crlf_double && (!removal_end || crlf_double < removal_end)) {
                removal_end = crlf_double + 4;
            }

            if (removal_end) {
                memmove(start, removal_end, strlen(removal_end) + 1);
            } else {
                *start = '\0';
            }
            break;
        }
    }
}

static void strip_leading_labels(char *text) {
    static const char *const labels[] = {"answer:",      "final answer:", "response:",
                                         "final:",       "reply:",        "output:",
                                         "result:"};
    char *start = NULL;

    if (!text) {
        return;
    }

    trim_leading_whitespace(text);
    start = text;

    for (size_t i = 0; i < (sizeof(labels) / sizeof(labels[0])); ++i) {
        size_t label_len = strlen(labels[i]);
        if (strncasecmp(start, labels[i], label_len) == 0) {
            char *after = start + label_len;
            while (*after && isspace((unsigned char)*after)) {
                after++;
            }
            memmove(start, after, strlen(after) + 1);
            break;
        }
    }
}

static char *find_name_label(char *text, const char *name, char **after_label) {
    size_t name_len = 0;
    char *cursor = NULL;

    if (after_label) {
        *after_label = NULL;
    }

    if (!text || !name || !*name) {
        return NULL;
    }

    name_len = strlen(name);
    cursor = text;

    while ((cursor = strcasestr(cursor, name)) != NULL) {
        char *next = cursor + name_len;

        if (cursor != text) {
            unsigned char prev = (unsigned char)cursor[-1];
            if (isalnum(prev) || prev == '_') {
                cursor += name_len;
                continue;
            }
        }

        while (*next && isspace((unsigned char)*next)) {
            next++;
        }

        if (*next == '(') {
            int depth = 1;
            next++;
            while (*next && depth > 0) {
                if (*next == '(') {
                    depth++;
                } else if (*next == ')') {
                    depth--;
                }
                next++;
            }
            while (*next && isspace((unsigned char)*next)) {
                next++;
            }
        }

        if (*next == ':') {
            char *content = next + 1;
            while (*content && isspace((unsigned char)*content)) {
                content++;
            }
            if (after_label) {
                *after_label = content;
            }
            return cursor;
        }

        cursor += name_len;
    }

    return NULL;
}

static void drop_text_before_name_label(char *text, const char *name) {
    char *after_label = NULL;
    char *label_start = find_name_label(text, name, &after_label);

    if (label_start && label_start != text) {
        memmove(text, label_start, strlen(label_start) + 1);
    }
}

static void strip_leading_name_label(char *text, const char *name) {
    char *after_label = NULL;
    char *label_start = find_name_label(text, name, &after_label);

    if (label_start == text && after_label) {
        memmove(text, after_label, strlen(after_label) + 1);
    }
}

static void sanitize_model_response(char *response, const char *participant_name,
                                    const char *display_label, const char *model_name) {
    const char *labels[3];
    size_t label_count = 0;

    if (!response) {
        return;
    }

    if (participant_name && *participant_name) {
        labels[label_count++] = participant_name;
    }
    if (display_label && *display_label) {
        int already_present = 0;
        for (size_t i = 0; i < label_count; ++i) {
            if (strcasecmp(labels[i], display_label) == 0) {
                already_present = 1;
                break;
            }
        }
        if (!already_present) {
            labels[label_count++] = display_label;
        }
    }
    if (model_name && *model_name) {
        int already_present = 0;
        for (size_t i = 0; i < label_count; ++i) {
            if (strcasecmp(labels[i], model_name) == 0) {
                already_present = 1;
                break;
            }
        }
        if (!already_present) {
            labels[label_count++] = model_name;
        }
    }

    remove_tagged_section(response, "<thinking>", "</thinking>");
    remove_tagged_section(response, "<think>", "</think>");
    remove_tagged_section(response, "<analysis>", "</analysis>");
    remove_tagged_section(response, "<scratchpad>", "</scratchpad>");
    remove_tagged_section(response, "[thinking]", "[/thinking]");
    remove_tagged_section(response, "[think]", "[/think]");
    remove_tagged_section(response, "{thinking}", "{/thinking}");
    remove_tagged_section(response, "{think}", "{/think}");

    trim_leading_whitespace(response);
    for (size_t i = 0; i < label_count; ++i) {
        drop_text_before_name_label(response, labels[i]);
    }
    remove_leading_metadata_block(response);
    trim_leading_whitespace(response);
    for (size_t i = 0; i < label_count; ++i) {
        strip_leading_name_label(response, labels[i]);
        trim_leading_whitespace(response);
    }
    strip_leading_labels(response);
    trim_leading_whitespace(response);
    trim_trailing_whitespace(response);
}

static char *duplicate_choice_content(json_object *parsed_json) {
    json_object *choices = NULL;

    if (!parsed_json) {
        return NULL;
    }

    if (!json_object_object_get_ex(parsed_json, "choices", &choices) ||
        json_object_get_type(choices) != json_type_array) {
        return NULL;
    }

    size_t choice_count = json_object_array_length(choices);
    for (size_t i = 0; i < choice_count; ++i) {
        json_object *choice = json_object_array_get_idx(choices, i);
        json_object *message = NULL;
        json_object *delta = NULL;
        json_object *content_obj = NULL;
        const char *content_str = NULL;

        if (choice && json_object_object_get_ex(choice, "message", &message) &&
            json_object_object_get_ex(message, "content", &content_obj) &&
            json_object_is_type(content_obj, json_type_string)) {
            content_str = json_object_get_string(content_obj);
        } else if (choice && json_object_object_get_ex(choice, "delta", &delta) &&
                   json_object_object_get_ex(delta, "content", &content_obj) &&
                   json_object_is_type(content_obj, json_type_string)) {
            content_str = json_object_get_string(content_obj);
        }

        if (content_str && *content_str) {
            return strdup(content_str);
        }
    }

    return NULL;
}

static char *parse_streaming_chunks(const char *payload) {
    char *combined = NULL;
    const char *cursor = payload;

    while (cursor && *cursor) {
        const char *data_marker = strstr(cursor, "data:");
        const char *line_end = NULL;
        size_t fragment_len = 0;

        if (!data_marker) {
            break;
        }

        data_marker += 5; /* skip "data:" */
        while (*data_marker == ' ' || *data_marker == '\t') {
            data_marker++;
        }

        line_end = strchr(data_marker, '\n');
        if (line_end) {
            fragment_len = (size_t)(line_end - data_marker);
        } else {
            fragment_len = strlen(data_marker);
        }

        while (fragment_len > 0 && (data_marker[fragment_len - 1] == '\r')) {
            fragment_len--;
        }

        cursor = line_end ? line_end + 1 : data_marker + fragment_len;
        if (fragment_len == 0) {
            continue;
        }

        if (fragment_len == 6 && strncmp(data_marker, "[DONE]", 6) == 0) {
            continue;
        }

        char *fragment = strndup(data_marker, fragment_len);
        json_object *parsed_fragment = NULL;
        json_object *error_obj = NULL;
        char *piece = NULL;

        if (!fragment) {
            free(combined);
            return NULL;
        }

        parsed_fragment = json_tokener_parse(fragment);
        free(fragment);
        if (!parsed_fragment) {
            continue;
        }

        if (json_object_object_get_ex(parsed_fragment, "error", &error_obj)) {
            const char *error_msg = json_object_get_string(error_obj);
            if (error_msg) {
                fprintf(stderr, "Error from AI server: %s\n", error_msg);
            }
            json_object_put(parsed_fragment);
            free(combined);
            return NULL;
        }

        piece = duplicate_choice_content(parsed_fragment);
        json_object_put(parsed_fragment);
        if (!piece) {
            continue;
        }

        char *new_combined = append_to_history(combined, piece);
        free(piece);
        if (!new_combined) {
            free(combined);
            return NULL;
        }
        combined = new_combined;
    }

    return combined;
}

static char *parse_ollama_response(const char *json_string) {
    struct json_object *parsed_json = NULL;
    struct json_object *response_obj = NULL;
    struct json_object *error_obj = NULL;
    char *response_text = NULL;

    if (!json_string) {
        return NULL;
    }

    parsed_json = json_tokener_parse(json_string);
    if (!parsed_json) {
        return parse_streaming_chunks(json_string);
    }

    if (json_object_object_get_ex(parsed_json, "error", &error_obj)) {
        const char *error_msg = json_object_get_string(error_obj);
        if (error_msg) {
            fprintf(stderr, "Error from AI server: %s\n", error_msg);
        }
    } else {
        response_text = duplicate_choice_content(parsed_json);
        if (!response_text &&
            json_object_object_get_ex(parsed_json, "response", &response_obj)) {
            const char *response_str = json_object_get_string(response_obj);
            if (response_str) {
                response_text = strdup(response_str);
            }
        }
    }

    json_object_put(parsed_json);
    return response_text;
}

static char *get_ai_response(json_object *messages_array, const char *model_name,
                             const char *participant_name, const char *display_label,
                             const char *ollama_url, int web_search_enabled) {
    CURL *curl = NULL;
    char *response = NULL;
    struct MemoryStruct chunk = {.memory = malloc(1), .size = 0};

    if (!chunk.memory) {
        fprintf(stderr, "Failed to allocate memory for response buffer.\n");
        return NULL;
    }

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (curl) {
        json_object *jobj = json_object_new_object();
        struct curl_slist *headers = NULL;
        /* *** MODIFIED FOR OPEN WEBUI *** */
        const char *api_key = get_webui_key();
        char auth_header[512];

        json_object_object_add(jobj, "model", json_object_new_string(model_name));
        json_object_object_add(jobj, "stream", json_object_new_boolean(1));
        json_object_object_add(jobj, "messages", json_object_get(messages_array));
        if (web_search_enabled) {
            json_object *features = json_object_new_object();
            json_object_object_add(features, "web_search", json_object_new_boolean(1));
            json_object_object_add(features, "image_generation", json_object_new_boolean(0));
            json_object_object_add(features, "code_interpreter", json_object_new_boolean(0));
            json_object_object_add(jobj, "features", features);
        }

        const char *json_payload = json_object_to_json_string(jobj);
        headers = curl_slist_append(NULL, "Content-Type: application/json");

        /* *** MODIFIED FOR OPEN WEBUI *** */
        if (api_key) {
            snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", api_key);
            headers = curl_slist_append(headers, auth_header);
        }

        curl_easy_setopt(curl, CURLOPT_URL, ollama_url);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);

        fprintf(stdout, "Requesting response from model '%s'...\n", model_name);
        CURLcode res = curl_easy_perform(curl);
        if (res == CURLE_OK) {
            response = parse_ollama_response(chunk.memory);
            sanitize_model_response(response, participant_name, display_label, model_name);
        } else {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }

        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
        json_object_put(jobj);
    }
    curl_global_cleanup();
    free(chunk.memory);
    return response;
}

static char *append_to_history(char *history, const char *text) {
    size_t old_len = history ? strlen(history) : 0;
    size_t text_len = strlen(text);
    char *new_history = realloc(history, old_len + text_len + 1);
    if (!new_history) {
        fprintf(stderr, "Failed to reallocate memory for history.\n");
        free(history);
        return NULL;
    }
    memcpy(new_history + old_len, text, text_len + 1);
    return new_history;
}

static char *build_webui_models_url(const char *ollama_url) {
    const char *marker = "/ollama/api/";
    const char *found = NULL;
    size_t base_len = 0;
    const char *suffix = "/api/models";

    if (!ollama_url) {
        return NULL;
    }

    found = strstr(ollama_url, marker);
    if (found) {
        base_len = (size_t)(found - ollama_url);
    } else {
        const char *scheme = strstr(ollama_url, "://");
        const char *path = NULL;
        if (scheme) {
            path = strchr(scheme + 3, '/');
        } else {
            path = strchr(ollama_url, '/');
        }
        if (path) {
            base_len = (size_t)(path - ollama_url);
        } else {
            base_len = strlen(ollama_url);
        }
    }

    char *result = malloc(base_len + strlen(suffix) + 1);
    if (!result) {
        return NULL;
    }
    memcpy(result, ollama_url, base_len);
    strcpy(result + base_len, suffix);
    return result;
}

static char *build_ollama_tags_url(const char *ollama_url) {
    const char *suffix = "/tags";
    const char *generate = "generate";
    size_t url_len = strlen(ollama_url);
    size_t generate_len = strlen(generate);
    char *result = NULL;

    if (url_len >= generate_len && strcmp(ollama_url + url_len - generate_len, generate) == 0) {
        size_t base_len = url_len - generate_len;
        result = malloc(base_len + strlen(suffix) + 1);
        if (!result) {
            return NULL;
        }
        memcpy(result, ollama_url, base_len);
        if (base_len > 0 && result[base_len - 1] == '/') {
            strcpy(result + base_len, suffix + 1);
        } else {
            strcpy(result + base_len, suffix);
        }
        return result;
    }

    int needs_slash = (url_len == 0 || ollama_url[url_len - 1] != '/');
    result = malloc(url_len + needs_slash + strlen(suffix) + 1);
    if (!result) {
        return NULL;
    }
    strcpy(result, ollama_url);
    if (needs_slash) {
        strcat(result, "/");
    }
    strcat(result, suffix + 1);
    return result;
}

static int fetch_models_via_url(const char *models_url, json_object **out_json, char **error_out) {
    struct MemoryStruct chunk = {.memory = NULL, .size = 0};
    CURL *curl = NULL;
    CURLcode res = CURLE_OK;
    json_object *parsed = NULL;
    json_object *models_array = NULL;
    json_object *result = NULL;
    json_object *list = NULL;
    long http_status = 0;

    if (error_out) {
        *error_out = NULL;
    }
    if (!models_url || !out_json) {
        if (error_out) {
            *error_out = strdup("Invalid models URL.");
        }
        return -1;
    }

    *out_json = NULL;
    chunk.memory = malloc(1);
    if (!chunk.memory) {
        if (error_out) {
            *error_out = strdup("Failed to allocate response buffer.");
        }
        return -1;
    }
    chunk.size = 0;

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (!curl) {
        free(chunk.memory);
        if (error_out) {
            *error_out = strdup("Unable to initialise CURL.");
        }
        curl_global_cleanup();
        return -1;
    }

    const char *api_key = get_webui_key();
    char auth_header[512];
    struct curl_slist *headers = NULL;

    if (api_key) {
        snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", api_key);
        headers = curl_slist_append(headers, auth_header);
    }

    curl_easy_setopt(curl, CURLOPT_URL, models_url);
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    if (headers) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }

    res = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    curl_global_cleanup();

    if (res != CURLE_OK || http_status < 200 || http_status >= 300) {
        if (res != CURLE_OK) {
            const char *error_str = curl_easy_strerror(res);
            fprintf(stderr,
                    "Model discovery request to %s failed: %s (curl code %d).\n",
                    models_url, error_str, res);
            fprintf(stderr,
                    "Tip: Run `curl%s \"%s\"` from the aiChat host to reproduce the failure.\n",
                    api_key ? " -H \"Authorization: Bearer <YOUR_WEBUI_API_KEY>\"" : "",
                    models_url);
        } else {
            fprintf(stderr,
                    "Model discovery request to %s returned HTTP %ld.\n",
                    models_url, http_status);
            fprintf(stderr,
                    "Tip: Run `curl%s -i \"%s\"` to inspect the response directly.\n",
                    api_key ? " -H \"Authorization: Bearer <YOUR_WEBUI_API_KEY>\"" : "",
                    models_url);
        }

        free(chunk.memory);
        if (error_out) {
            if (res != CURLE_OK) {
                char buffer[192];
                snprintf(buffer, sizeof(buffer),
                         "Failed to contact Open WebUI for model list (curl error: %s).",
                         curl_easy_strerror(res));
                *error_out = strdup(buffer);
            } else {
                char buffer[160];
                snprintf(buffer, sizeof(buffer),
                         "Open WebUI returned status %ld while listing models.", http_status);
                *error_out = strdup(buffer);
            }
        }
        return -1;
    }

    parsed = json_tokener_parse(chunk.memory);
    if (parsed && json_object_is_type(parsed, json_type_object)) {
        json_object *field = NULL;
        if (json_object_object_get_ex(parsed, "models", &field)) {
            models_array = field;
        } else if (json_object_object_get_ex(parsed, "data", &field)) {
            models_array = field;
        } else if (json_object_object_get_ex(parsed, "items", &field)) {
            models_array = field;
        } else if (json_object_object_get_ex(parsed, "list", &field)) {
            models_array = field;
        }
    } else if (parsed && json_object_is_type(parsed, json_type_array)) {
        models_array = parsed;
    }

    if (!models_array || !json_object_is_type(models_array, json_type_array)) {
        fprintf(stderr,
                "Model discovery request to %s returned an unexpected payload (first 200 bytes shown): %.200s\n",
                models_url,
                chunk.memory ? chunk.memory : "<empty response>");
        fprintf(stderr,
                "Tip: Run `curl%s \"%s\"` and review the JSON body.\n",
                api_key ? " -H \"Authorization: Bearer <YOUR_WEBUI_API_KEY>\"" : "",
                models_url);
        free(chunk.memory);
        if (parsed) {
            json_object_put(parsed);
        }
        if (error_out) {
            *error_out = strdup("Unexpected response from Open WebUI while listing models.");
        }
        return -1;
    }

    list = json_object_new_array();
    if (!list) {
        free(chunk.memory);
        json_object_put(parsed);
        if (error_out) {
            *error_out = strdup("Failed to allocate models array.");
        }
        return -1;
    }

    size_t array_len = json_object_array_length(models_array);
    for (size_t i = 0; i < array_len; ++i) {
        json_object *item = json_object_array_get_idx(models_array, i);
        const char *model_value = NULL;
        const char *name_value = NULL;

        if (!item) {
            continue;
        }

        if (json_object_is_type(item, json_type_object)) {
            json_object *field = NULL;
            if (json_object_object_get_ex(item, "model", &field) && field &&
                json_object_get_type(field) == json_type_string) {
                model_value = json_object_get_string(field);
            }
            if (!model_value && json_object_object_get_ex(item, "provider_model_id", &field) && field &&
                json_object_get_type(field) == json_type_string) {
                model_value = json_object_get_string(field);
            }
            if (!model_value && json_object_object_get_ex(item, "id", &field) && field &&
                json_object_get_type(field) == json_type_string) {
                model_value = json_object_get_string(field);
            }
            if (json_object_object_get_ex(item, "name", &field) && field &&
                json_object_get_type(field) == json_type_string) {
                name_value = json_object_get_string(field);
            } else if (json_object_object_get_ex(item, "display_name", &field) && field &&
                       json_object_get_type(field) == json_type_string) {
                name_value = json_object_get_string(field);
            } else if (json_object_object_get_ex(item, "title", &field) && field &&
                       json_object_get_type(field) == json_type_string) {
                name_value = json_object_get_string(field);
            } else if (json_object_object_get_ex(item, "label", &field) && field &&
                       json_object_get_type(field) == json_type_string) {
                name_value = json_object_get_string(field);
            }
        } else if (json_object_is_type(item, json_type_string)) {
            model_value = json_object_get_string(item);
        }

        if (!model_value || !*model_value) {
            if (name_value && *name_value) {
                model_value = name_value;
            } else {
                continue;
            }
        }

        json_object *entry = json_object_new_object();
        if (!entry) {
            json_object_put(list);
            json_object_put(parsed);
            free(chunk.memory);
            if (error_out) {
                *error_out = strdup("Failed to allocate model entry.");
            }
            return -1;
        }

        const char *display = (name_value && *name_value) ? name_value : model_value;
        json_object_object_add(entry, "name", json_object_new_string(display));
        json_object_object_add(entry, "model", json_object_new_string(model_value));
        json_object_array_add(list, entry);
    }

    result = json_object_new_object();
    if (!result) {
        json_object_put(list);
        json_object_put(parsed);
        free(chunk.memory);
        if (error_out) {
            *error_out = strdup("Failed to prepare models payload.");
        }
        return -1;
    }

    json_object_object_add(result, "models", list);
    *out_json = result;

    json_object_put(parsed);
    free(chunk.memory);
    return 0;
}

static int fetch_available_models(const char *ollama_url, json_object **out_json, char **error_out) {
    char *webui_url = build_webui_models_url(ollama_url);
    char *fallback_url = build_ollama_tags_url(ollama_url);
    char *webui_error = NULL;
    char *fallback_error = NULL;
    int rc = -1;

    if (error_out) {
        *error_out = NULL;
    }

    if (webui_url) {
        rc = fetch_models_via_url(webui_url, out_json, &webui_error);
        free(webui_url);
        if (rc == 0) {
            if (webui_error) {
                free(webui_error);
            }
            if (fallback_url) {
                free(fallback_url);
            }
            return 0;
        }
        if (webui_error) {
            fprintf(stderr, "Open WebUI model probe failed: %s\n", webui_error);
        }
    } else {
        webui_error = strdup("Failed to prepare Open WebUI models URL.");
    }

    if (fallback_url) {
        fprintf(stderr, "Falling back to legacy Ollama tags endpoint for model discovery.\n");
        rc = fetch_models_via_url(fallback_url, out_json, &fallback_error);
        free(fallback_url);
        if (rc == 0) {
            if (webui_error) {
                free(webui_error);
            }
            if (fallback_error) {
                free(fallback_error);
            }
            return 0;
        }
    }

    if (error_out) {
        if (fallback_error) {
            *error_out = fallback_error;
            fallback_error = NULL;
        } else if (webui_error) {
            *error_out = webui_error;
            webui_error = NULL;
        } else {
            *error_out = strdup("Unable to retrieve model list.");
        }
    }

    if (webui_error) {
        free(webui_error);
    }
    if (fallback_error) {
        free(fallback_error);
    }
    return -1;
}

static void handle_models_request(int client_fd, const char *ollama_url) {
    json_object *payload = NULL;
    char *error_message = NULL;

    if (fetch_available_models(ollama_url, &payload, &error_message) == 0 && payload) {
        const char *json_payload = json_object_to_json_string_ext(payload, JSON_C_TO_STRING_PLAIN);
        send_http_response(client_fd, "200 OK", "application/json", json_payload);
        json_object_put(payload);
    } else {
        const char *message = error_message ? error_message : "Unable to retrieve model list.";
        send_http_error(client_fd, "502 Bad Gateway", message);
    }

    if (error_message) {
        free(error_message);
    }
}

static void handle_diagnostics_request(int client_fd, const char *ollama_url) {
    json_object *payload = json_object_new_object();
    char *models_url = NULL;
    char *fallback_url = NULL;
    const char *api_endpoint = ollama_url ? ollama_url : "";
    const char *api_key = get_webui_key();

    if (!payload) {
        send_http_error(client_fd, "500 Internal Server Error", "Unable to prepare diagnostics payload.");
        return;
    }

    json_object_object_add(payload, "webuiEndpoint", json_object_new_string(api_endpoint));

    models_url = build_webui_models_url(ollama_url);
    if (models_url) {
        json_object_object_add(payload, "modelsUrl", json_object_new_string(models_url));
        free(models_url);
    } else {
        json_object_object_add(payload, "modelsUrl", json_object_new_string(""));
    }

    fallback_url = build_ollama_tags_url(ollama_url);
    if (fallback_url) {
        json_object_object_add(payload, "fallbackUrl", json_object_new_string(fallback_url));
        free(fallback_url);
    } else {
        json_object_object_add(payload, "fallbackUrl", json_object_new_string(""));
    }

    json_object_object_add(payload, "usesApiKey", json_object_new_boolean(api_key != NULL));

    const char *json_payload = json_object_to_json_string_ext(payload, JSON_C_TO_STRING_PLAIN);
    send_http_response(client_fd, "200 OK", "application/json", json_payload);
    json_object_put(payload);
}

static const char *lookup_display_model(json_object *models_array, const char *identifier) {
    size_t array_len = 0;

    if (!models_array || !identifier || !*identifier) {
        return NULL;
    }

    if (!json_object_is_type(models_array, json_type_array)) {
        return NULL;
    }

    array_len = json_object_array_length(models_array);
    for (size_t i = 0; i < array_len; ++i) {
        json_object *item = json_object_array_get_idx(models_array, i);
        const char *model_value = NULL;
        const char *name_value = NULL;

        if (!item || !json_object_is_type(item, json_type_object)) {
            continue;
        }

        json_object *field = NULL;
        if (json_object_object_get_ex(item, "model", &field) && field) {
            model_value = json_object_get_string(field);
        }
        if (json_object_object_get_ex(item, "name", &field) && field) {
            name_value = json_object_get_string(field);
        }

        if (model_value && strcmp(model_value, identifier) == 0) {
            if (name_value && *name_value) {
                return name_value;
            }
            return model_value;
        }
        if (name_value && strcmp(name_value, identifier) == 0) {
            return name_value;
        }
    }

    return NULL;
}

static void ensure_participant_display_models(struct Participant *participants, size_t participant_count,
                                              json_object *models_payload) {
    json_object *models_array = NULL;

    if (!participants || participant_count == 0) {
        return;
    }

    if (models_payload && json_object_is_type(models_payload, json_type_object)) {
        json_object_object_get_ex(models_payload, "models", &models_array);
    }

    for (size_t i = 0; i < participant_count; ++i) {
        trim_leading_whitespace(participants[i].display_model);
        trim_trailing_whitespace(participants[i].display_model);
        if (participants[i].display_model[0] != '\0') {
            continue;
        }

        const char *display = lookup_display_model(models_array, participants[i].model);
        if (!display || !*display) {
            display = participants[i].model;
        }

        strncpy(participants[i].display_model, display, MAX_MODEL_LENGTH - 1);
        participants[i].display_model[MAX_MODEL_LENGTH - 1] = '\0';
        trim_leading_whitespace(participants[i].display_model);
        trim_trailing_whitespace(participants[i].display_model);
    }
}

static int run_conversation(const char *topic, int turns, struct Participant *participants,
                            size_t participant_count, const char *ollama_url, int search_enabled,
                            message_callback on_message, void *callback_data, json_object **out_json,
                            char **error_out) {
    char *conversation_history = NULL;
    json_object *result_messages = NULL;
    json_object *participants_json = NULL;
    json_object *request_messages = NULL;
    json_object *result = NULL;

    *out_json = NULL;
    if (error_out) {
        *error_out = NULL;
    }

    conversation_history = strdup(SYSTEM_PROMPT);
    if (!conversation_history) {
        if (error_out) {
            *error_out = strdup("Failed to allocate conversation history.");
        }
        return -1;
    }

    conversation_history = append_to_history(conversation_history, "USER: ");
    if (!conversation_history) {
        if (error_out) {
            *error_out = strdup("Failed to build conversation history.");
        }
        return -1;
    }

    conversation_history = append_to_history(conversation_history, topic);
    if (!conversation_history) {
        if (error_out) {
            *error_out = strdup("Failed to build conversation history.");
        }
        return -1;
    }

    request_messages = json_object_new_array();
    result_messages = json_object_new_array();
    participants_json = json_object_new_array();
    if (!request_messages || !result_messages || !participants_json) {
        if (error_out) {
            *error_out = strdup("Failed to allocate JSON structures.");
        }
        goto fail;
    }

    {
        json_object *system_message = json_object_new_object();
        json_object *topic_message = json_object_new_object();

        if (!system_message || !topic_message) {
            if (system_message) {
                json_object_put(system_message);
            }
            if (topic_message) {
                json_object_put(topic_message);
            }
            if (error_out) {
                *error_out = strdup("Failed to initialize conversation messages.");
            }
            goto fail;
        }

        json_object_object_add(system_message, "role", json_object_new_string("system"));
        json_object_object_add(system_message, "content", json_object_new_string(SYSTEM_PROMPT));
        json_object_array_add(request_messages, system_message);

        json_object_object_add(topic_message, "role", json_object_new_string("user"));
        json_object_object_add(topic_message, "content", json_object_new_string(topic ? topic : ""));
        json_object_array_add(request_messages, topic_message);
    }

    for (size_t p = 0; p < participant_count; ++p) {
        json_object *participant_obj = json_object_new_object();
        if (!participant_obj) {
            if (error_out) {
                *error_out = strdup("Failed to allocate participant JSON.");
            }
            goto fail;
        }
        json_object_object_add(participant_obj, "name", json_object_new_string(participants[p].name));
        json_object_object_add(participant_obj, "model", json_object_new_string(participants[p].model));
        if (participants[p].display_model[0] != '\0') {
            json_object_object_add(participant_obj, "displayModel",
                                   json_object_new_string(participants[p].display_model));
        }
        json_object_array_add(participants_json, participant_obj);
    }

    for (int turn = 0; turn < turns; ++turn) {
        for (size_t idx = 0; idx < participant_count; ++idx) {
            char label[128];
            char *response = NULL;
            json_object *message = NULL;
            json_object *prompt_message = NULL;
            json_object *assistant_message = NULL;
            size_t prompt_index = 0;
            char prompt_content[128];

            snprintf(label, sizeof(label), "\n\n%s:", participants[idx].name);
            conversation_history = append_to_history(conversation_history, label);
            if (!conversation_history) {
                if (error_out) {
                    *error_out = strdup("Failed to build conversation history.");
                }
                goto fail;
            }

            prompt_message = json_object_new_object();
            if (!prompt_message) {
                if (error_out) {
                    *error_out = strdup("Failed to allocate prompt message.");
                }
                goto fail;
            }

            snprintf(prompt_content, sizeof(prompt_content), "%s:", participants[idx].name);
            json_object_object_add(prompt_message, "role", json_object_new_string("user"));
            json_object_object_add(prompt_message, "content", json_object_new_string(prompt_content));
            prompt_index = json_object_array_length(request_messages);
            json_object_array_add(request_messages, prompt_message);

            // Only enable web search for the first participant on the first turn.
            int is_first_request = (turn == 0 && idx == 0);
            int use_web_search = search_enabled && is_first_request;

            response = get_ai_response(request_messages, participants[idx].model,
                                       participants[idx].name, participants[idx].display_model,
                                       ollama_url, use_web_search);
            if (!response) {
                json_object_array_del_idx(request_messages, prompt_index, 1);
                if (error_out) {
                    char buffer[256];
                    snprintf(buffer, sizeof(buffer), "Model '%.*s' failed to respond.",
                             (int)(sizeof(buffer) - 40), participants[idx].model);
                    *error_out = strdup(buffer);
                }
                goto fail;
            }

            // Remove the temporary prompt message before adding the assistant response.
            json_object_array_del_idx(request_messages, prompt_index, 1);

            assistant_message = json_object_new_object();
            if (!assistant_message) {
                free(response);
                if (error_out) {
                    *error_out = strdup("Failed to allocate assistant message.");
                }
                goto fail;
            }

            conversation_history = append_to_history(conversation_history, response);
            if (!conversation_history) {
                free(response);
                if (error_out) {
                    *error_out = strdup("Failed to build conversation history.");
                }
                goto fail;
            }

            message = json_object_new_object();
            if (!message) {
                free(response);
                if (error_out) {
                    *error_out = strdup("Failed to allocate message JSON.");
                }
                goto fail;
            }

            json_object_object_add(message, "turn", json_object_new_int(turn + 1));
            json_object_object_add(message, "participantIndex", json_object_new_int((int)idx));
            json_object_object_add(message, "name", json_object_new_string(participants[idx].name));
            json_object_object_add(message, "model", json_object_new_string(participants[idx].model));
            if (participants[idx].display_model[0] != '\0') {
                json_object_object_add(message, "displayModel",
                                       json_object_new_string(participants[idx].display_model));
            }
            json_object_object_add(message, "text", json_object_new_string(response));
            json_object_array_add(result_messages, message);

            json_object_object_add(assistant_message, "role", json_object_new_string("assistant"));
            json_object_object_add(assistant_message, "content", json_object_new_string(response));
            json_object_array_add(request_messages, assistant_message);

            if (on_message) {
                json_object_get(message);
                if (on_message(message, callback_data) != 0) {
                    json_object_put(message);
                    free(response);
                    if (error_out && (!*error_out)) {
                        *error_out = strdup("Failed to stream message.");
                    }
                    goto fail;
                }
                json_object_put(message);
            }

            free(response);
        }
    }
    result = json_object_new_object();
    if (!result) {
        if (error_out) {
            *error_out = strdup("Failed to allocate result JSON.");
        }
        goto fail;
    }

    json_object_object_add(result, "topic", json_object_new_string(topic));
    json_object_object_add(result, "turns", json_object_new_int(turns));
    json_object_object_add(result, "participants", participants_json);
    json_object_object_add(result, "messages", result_messages);
    json_object_object_add(result, "history", json_object_new_string(conversation_history));
    json_object_object_add(result, "searchEnabled", json_object_new_boolean(search_enabled));

    json_object_put(request_messages);
    free(conversation_history);
    *out_json = result;
    return 0;

fail:
    if (result_messages) {
        json_object_put(result_messages);
    }
    if (participants_json) {
        json_object_put(participants_json);
    }
    if (request_messages) {
        json_object_put(request_messages);
    }
    if (result) {
        json_object_put(result);
    }
    free(conversation_history);
    return -1;
}

static void send_http_response(int client_fd, const char *status, const char *content_type,
                               const char *body) {
    char header[512];
    size_t body_length = body ? strlen(body) : 0;
    int header_len = snprintf(header, sizeof(header),
                              "HTTP/1.1 %s\r\n"
                              "Content-Type: %s\r\n"
                              "Content-Length: %zu\r\n"
                              "Access-Control-Allow-Origin: *\r\n"
                              "Connection: close\r\n\r\n",
                              status, content_type, body_length);
    if (header_len < 0 || (size_t)header_len >= sizeof(header)) {
        return;
    }

    send(client_fd, header, (size_t)header_len, 0);
    if (body_length > 0) {
        send(client_fd, body, body_length, 0);
    }
}

static void send_http_error(int client_fd, const char *status, const char *message) {
    json_object *obj = json_object_new_object();
    const char *payload = NULL;

    if (!obj) {
        send_http_response(client_fd, status, "text/plain; charset=UTF-8", message);
        return;
    }

    json_object_object_add(obj, "error", json_object_new_string(message));
    payload = json_object_to_json_string(obj);
    send_http_response(client_fd, status, "application/json", payload);
    json_object_put(obj);
}

static int send_all(int client_fd, const char *data, size_t length) {
    size_t total_sent = 0;

    while (total_sent < length) {
        ssize_t written = send(client_fd, data + total_sent, length - total_sent, 0);
        if (written <= 0) {
            return -1;
        }
        total_sent += (size_t)written;
    }

    return 0;
}

static int send_chunked_header(int client_fd, const char *status, const char *content_type) {
    char header[512];
    int header_len = snprintf(header, sizeof(header),
                              "HTTP/1.1 %s\r\n"
                              "Content-Type: %s\r\n"
                              "Transfer-Encoding: chunked\r\n"
                              "Cache-Control: no-cache\r\n"
                              "Access-Control-Allow-Origin: *\r\n"
                              "Connection: close\r\n\r\n",
                              status, content_type);

    if (header_len < 0 || (size_t)header_len >= sizeof(header)) {
        return -1;
    }

    return send_all(client_fd, header, (size_t)header_len);
}

static int send_json_chunk(int client_fd, json_object *obj) {
    const char *json = json_object_to_json_string_ext(obj, JSON_C_TO_STRING_PLAIN);
    char size_buffer[32];
    int size_len = 0;
    size_t json_len = 0;

    if (!json) {
        return -1;
    }

    json_len = strlen(json);
    size_len = snprintf(size_buffer, sizeof(size_buffer), "%zx\r\n", json_len + 1);
    if (size_len < 0 || (size_t)size_len >= sizeof(size_buffer)) {
        return -1;
    }

    if (send_all(client_fd, size_buffer, (size_t)size_len) != 0) {
        return -1;
    }

    if (send_all(client_fd, json, json_len) != 0) {
        return -1;
    }

    if (send_all(client_fd, "\n", 1) != 0) {
        return -1;
    }

    if (send_all(client_fd, "\r\n", 2) != 0) {
        return -1;
    }

    return 0;
}

static int finish_chunked_response(int client_fd) {
    return send_all(client_fd, "0\r\n\r\n", 5);
}

static int send_stream_error_event(int client_fd, const char *message) {
    json_object *event = json_object_new_object();
    int result = -1;

    if (!event) {
        return -1;
    }

    json_object_object_add(event, "type", json_object_new_string("error"));
    json_object_object_add(event, "message",
                           json_object_new_string(message ? message : "Conversation failed."));

    result = send_json_chunk(client_fd, event);
    json_object_put(event);
    return result;
}

struct StreamContext {
    int client_fd;
    int failed;
};

static int stream_message_callback(json_object *message, void *user_data) {
    struct StreamContext *ctx = (struct StreamContext *)user_data;
    json_object *event = NULL;
    int rc = -1;

    if (!ctx || ctx->failed) {
        return -1;
    }

    event = json_object_new_object();
    if (!event) {
        ctx->failed = 1;
        return -1;
    }

    json_object_object_add(event, "type", json_object_new_string("message"));
    json_object_object_add(event, "message", json_object_get(message));

    rc = send_json_chunk(ctx->client_fd, event);
    if (rc != 0) {
        ctx->failed = 1;
    }

    json_object_put(event);
    return rc;
}

static const char *get_html_page(void) {
    return "<!DOCTYPE html>\n"
           "<html lang=\"en\">\n"
           "<head>\n"
           "  <meta charset=\"UTF-8\" />\n"
           "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
           "  <title>aiChat Arena</title>\n"
           "  <style>\n"
           "    :root { color-scheme: dark; }\n"
           "    * { box-sizing: border-box; }\n"
           "    body { margin: 0; padding: clamp(1.5rem, 3vw, 3rem); font-family: 'Segoe UI', 'Orbitron', 'Roboto', sans-serif; background: radial-gradient(circle at 20% 20%, rgba(56, 189, 248, 0.15), transparent 55%), radial-gradient(circle at 80% 0%, rgba(168, 85, 247, 0.12), transparent 45%), #020617; color: #e2e8f0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; gap: 2.5rem; position: relative; overflow-x: hidden; }\n"
           "    body::before { content: ''; position: fixed; inset: -20vmax; background: conic-gradient(from 180deg at 50% 50%, rgba(56, 189, 248, 0.08), rgba(59, 130, 246, 0.16), rgba(168, 85, 247, 0.12), rgba(56, 189, 248, 0.08)); opacity: 0.65; filter: blur(120px); animation: auroraSpin 48s linear infinite; z-index: -2; pointer-events: none; }\n"
           "    body::after { content: ''; position: fixed; inset: -18vmax; background: radial-gradient(circle at 15% 25%, rgba(59, 130, 246, 0.22), transparent 55%), radial-gradient(circle at 85% 15%, rgba(168, 85, 247, 0.18), transparent 60%); opacity: 0.45; filter: blur(90px); animation: auroraPulse 32s ease-in-out infinite alternate; z-index: -3; pointer-events: none; }\n"
           "    @keyframes auroraSpin {\n"
           "      from { transform: rotate(0deg); }\n"
           "      to { transform: rotate(360deg); }\n"
           "    }\n"
           "    @keyframes auroraPulse {\n"
           "      0% { opacity: 0.35; transform: scale(0.95); }\n"
           "      50% { opacity: 0.55; transform: scale(1.05); }\n"
           "      100% { opacity: 0.35; transform: scale(1); }\n"
           "    }\n"
           "    .card { position: relative; width: min(960px, 100%); background: rgba(15, 23, 42, 0.75); border-radius: 24px; padding: clamp(1.5rem, 3vw, 2.5rem); border: 1px solid rgba(148, 163, 184, 0.25); box-shadow: 0 40px 80px rgba(2, 6, 23, 0.6); backdrop-filter: blur(18px); overflow: hidden; transform-style: preserve-3d; transition: transform 0.6s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.6s ease; }\n"
           "    .card::after { content: ''; position: absolute; inset: -40%; background: radial-gradient(circle at 30% 30%, rgba(56, 189, 248, 0.45), transparent 65%), radial-gradient(circle at 70% 10%, rgba(168, 85, 247, 0.35), transparent 60%); opacity: 0; transform: translate3d(0, 40px, 0) scale(0.95); filter: blur(40px); transition: opacity 0.6s ease, transform 0.6s cubic-bezier(0.22, 1, 0.36, 1); pointer-events: none; z-index: 0; }\n"
           "    .card > * { position: relative; z-index: 1; }\n"
           "    .card:hover { transform: translateY(-6px); box-shadow: 0 48px 120px rgba(2, 6, 23, 0.75); }\n"
           "    .card:hover::after { opacity: 1; transform: translate3d(0, 0, 0) scale(1.05); }\n"
           "    h1 { margin-bottom: 0.25rem; font-size: clamp(1.9rem, 3vw, 2.4rem); letter-spacing: 0.08em; text-transform: uppercase; color: #f8fafc; }\n"
           "    h2 { margin-top: 0; letter-spacing: 0.06em; text-transform: uppercase; color: #cbd5f5; }\n"
           "    p { margin-top: 0; color: rgba(226, 232, 240, 0.85); }\n"
           "    label { display: block; margin-top: 1.25rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: rgba(148, 163, 184, 0.9); }\n"
           "    input, select { width: 100%; padding: 0.75rem 1rem; margin-top: 0.5rem; border-radius: 12px; border: 1px solid rgba(148, 163, 184, 0.25); background: rgba(15, 23, 42, 0.6); color: #f8fafc; box-shadow: inset 0 0 0 rgba(15, 23, 42, 0.5); transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.3s ease; }\n"
           "    input:focus, select:focus { outline: none; border-color: rgba(94, 234, 212, 0.8); box-shadow: 0 0 0 3px rgba(94, 234, 212, 0.2); transform: translateY(-1px); }\n"
           "    input::placeholder { color: rgba(148, 163, 184, 0.6); }\n"
           "    .actions { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 1.5rem; }\n"
           "    .search-control { margin-top: 1.25rem; }\n"
           "    .form-toggle { display: flex; align-items: center; gap: 0.75rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: rgba(148, 163, 184, 0.88); transition: opacity 0.2s ease; }\n"
           "    .form-toggle.is-disabled { opacity: 0.6; }\n"
           "    .form-toggle span { text-transform: none; letter-spacing: 0.03em; color: rgba(226, 232, 240, 0.9); }\n"
           "    .form-toggle input { width: 1.2rem; height: 1.2rem; border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.35); background: rgba(15, 23, 42, 0.6); cursor: pointer; accent-color: #38bdf8; }\n"
           "    .form-toggle input:focus-visible { outline: none; box-shadow: 0 0 0 3px rgba(94, 234, 212, 0.25); }\n"
           "    .form-hint { margin-top: 0.35rem; font-size: 0.85rem; color: rgba(148, 163, 184, 0.78); letter-spacing: 0.02em; }\n"
           "    .form-hint.is-warning { color: rgba(248, 113, 113, 0.85); }\n"
           "    .form-hint.is-active { color: rgba(94, 234, 212, 0.88); }\n"
           "    button { position: relative; padding: 0.85rem 1.8rem; border: none; border-radius: 999px; background: linear-gradient(120deg, #22d3ee, #a855f7); background-size: 220% 220%; color: #0b1120; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; cursor: pointer; transition: transform 0.25s ease, box-shadow 0.25s ease, background-position 0.4s ease; box-shadow: 0 12px 26px rgba(168, 85, 247, 0.35); }\n"
           "    button::after { content: ''; position: absolute; inset: -2px; border-radius: inherit; border: 1px solid rgba(255, 255, 255, 0.25); opacity: 0; transition: opacity 0.3s ease, transform 0.3s ease; pointer-events: none; }\n"
           "    button:hover { transform: translateY(-2px); box-shadow: 0 20px 45px rgba(168, 85, 247, 0.45); background-position: 100% 50%; }\n"
           "    button:hover::after { opacity: 1; transform: scale(1.03); }\n"
           "    button:focus-visible { outline: none; box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.35); }\n"
           "    button:active { transform: translateY(0); }\n"
           "    .participants { margin-top: 1.5rem; display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); perspective: 1200px; }\n"
           "    .participant { position: relative; border-radius: 18px; padding: 1.25rem; border: 1px solid var(--participant-border, rgba(148, 163, 184, 0.45)); background: var(--participant-surface, rgba(15, 23, 42, 0.6)); color: #e2e8f0; box-shadow: 0 0 32px var(--participant-glow, rgba(15, 23, 42, 0.8)); overflow: hidden; backdrop-filter: blur(12px); transform-style: preserve-3d; background-size: 180% 180%; transition: transform 0.6s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.6s ease, border-color 0.3s ease; }\n"
           "    .participant::before { content: ''; position: absolute; inset: 0; background: var(--participant-pattern, rgba(94, 234, 212, 0.1)); opacity: 0.55; filter: blur(60px); z-index: 0; transition: opacity 0.6s ease, transform 0.6s ease; animation: participantDrift 24s linear infinite; }\n"
           "    .participant::after { content: ''; position: absolute; inset: 1px; border-radius: 16px; background: radial-gradient(circle at 20% 20%, var(--participant-highlight, rgba(255, 255, 255, 0.35)), transparent 55%); opacity: 0; mix-blend-mode: screen; transition: opacity 0.6s ease; z-index: 0; }\n"
           "    .participant > * { position: relative; z-index: 1; }\n"
           "    .participant:hover, .participant:focus-within { transform: translateY(-8px) rotate3d(1, -1, 0, 6deg); box-shadow: 0 32px 80px var(--participant-glow, rgba(15, 23, 42, 0.8)); }\n"
           "    .participant:hover::before, .participant:focus-within::before { opacity: 0.8; transform: scale(1.08); }\n"
           "    .participant:hover::after, .participant:focus-within::after { opacity: 0.85; }\n"
           "    @keyframes participantDrift {\n"
           "      0% { transform: scale(1) translate3d(0, 0, 0); }\n"
           "      50% { transform: scale(1.05) translate3d(-6px, 4px, 0); }\n"
           "      100% { transform: scale(1) translate3d(0, 0, 0); }\n"
           "    }\n"
           "    .participant button { margin-top: 1rem; width: fit-content; background: rgba(15, 23, 42, 0.65); color: #f8fafc; border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 12px; padding: 0.5rem 1rem; letter-spacing: 0.05em; text-transform: none; font-size: 0.85rem; transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease; }\n"
           "    .participant button::after { display: none; }\n"
           "    .participant button:hover, .participant button:focus-visible { color: #f87171; border-color: rgba(248, 113, 113, 0.8); box-shadow: 0 0 20px rgba(248, 113, 113, 0.25); transform: translateY(-1px); }\n"
           "    .status-bar { margin-top: 1.25rem; display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }\n"
           "    .connection-flag { display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.35rem 0.85rem; border-radius: 999px; border: 1px solid rgba(148, 163, 184, 0.3); background: rgba(15, 23, 42, 0.55); color: rgba(226, 232, 240, 0.8); font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase; transition: border-color 0.3s ease, background 0.3s ease, color 0.3s ease, box-shadow 0.3s ease; }\n"
           "    .connection-flag .connection-dot { width: 0.55rem; height: 0.55rem; border-radius: 999px; background: #fbbf24; box-shadow: 0 0 10px rgba(251, 191, 36, 0.4); transition: background 0.3s ease, box-shadow 0.3s ease; }\n"
           "    .connection-flag.connection-connected { border-color: rgba(94, 234, 212, 0.75); background: rgba(13, 148, 136, 0.28); color: rgba(204, 251, 241, 0.95); box-shadow: 0 8px 24px rgba(13, 148, 136, 0.35); }\n"
           "    .connection-flag.connection-connected .connection-dot { background: #34d399; box-shadow: 0 0 12px rgba(52, 211, 153, 0.6); }\n"
           "    .connection-flag.connection-disconnected { border-color: rgba(248, 113, 113, 0.85); background: rgba(153, 27, 27, 0.28); color: rgba(254, 202, 202, 0.95); box-shadow: 0 8px 24px rgba(248, 113, 113, 0.35); }\n"
           "    .connection-flag.connection-disconnected .connection-dot { background: #f87171; box-shadow: 0 0 12px rgba(248, 113, 113, 0.65); }\n"
           "    .connection-flag.connection-checking { border-color: rgba(148, 163, 184, 0.45); background: rgba(30, 41, 59, 0.5); color: rgba(226, 232, 240, 0.82); box-shadow: 0 6px 18px rgba(15, 23, 42, 0.45); }\n"
           "    .connection-flag.connection-checking .connection-dot { background: #fbbf24; box-shadow: 0 0 12px rgba(251, 191, 36, 0.5); }\n"
           "    .connection-flag .connection-text { letter-spacing: 0.08em; }\n"
           "    #status { font-weight: 600; color: #f97316; letter-spacing: 0.05em; opacity: 0; transform: translateY(-0.25rem); transition: opacity 0.35s ease, transform 0.35s ease; text-shadow: none; }\n"
           "    #status.status-active { opacity: 1; transform: translateY(0); }\n"
           "    #status.status-flash { animation: statusGlow 1.2s ease-out; }\n"
           "    @keyframes statusGlow {\n"
           "      0% { opacity: 0.6; text-shadow: 0 0 0 rgba(249, 115, 22, 0.75); }\n"
           "      45% { opacity: 1; text-shadow: 0 0 22px rgba(249, 115, 22, 0.85); }\n"
           "      100% { opacity: 1; text-shadow: none; }\n"
           "    }\n"
           "    #status:empty { display: none; }\n"
           "    .connection-diagnostics { margin-top: 0.75rem; border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 16px; background: rgba(15, 23, 42, 0.6); box-shadow: 0 18px 36px rgba(2, 6, 23, 0.4); overflow: hidden; backdrop-filter: blur(12px); transition: border-color 0.3s ease, box-shadow 0.3s ease; }\n"
           "    .connection-diagnostics summary { cursor: pointer; padding: 0.8rem 1rem; list-style: none; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: rgba(226, 232, 240, 0.85); display: flex; align-items: center; gap: 0.5rem; }\n"
           "    .connection-diagnostics summary::-webkit-details-marker { display: none; }\n"
           "    .connection-diagnostics summary::before { content: '\25B8'; display: inline-block; transition: transform 0.3s ease; }\n"
           "    .connection-diagnostics[open] summary::before { transform: rotate(90deg); }\n"
           "    .connection-diagnostics summary:focus-visible { outline: none; color: rgba(94, 234, 212, 0.95); }\n"
           "    .connection-diagnostics .diagnostics-content { padding: 0.75rem 1rem 1rem; border-top: 1px solid rgba(148, 163, 184, 0.25); display: grid; gap: 0.75rem; background: linear-gradient(160deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.75)); }\n"
           "    .connection-diagnostics pre { margin: 0; background: rgba(2, 6, 23, 0.65); border-radius: 12px; padding: 0.75rem 1rem; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.85rem; color: rgba(226, 232, 240, 0.92); overflow-x: auto; box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.25); }\n"
           "    .connection-diagnostics code { font-family: inherit; }\n"
           "    .connection-diagnostics.diagnostics-error { border-color: rgba(248, 113, 113, 0.7); box-shadow: 0 20px 45px rgba(248, 113, 113, 0.25); }\n"
           "    .connection-diagnostics.diagnostics-error summary { color: rgba(254, 202, 202, 0.95); }\n"
           "    #transcript { width: min(960px, 100%); }\n"
           "    .export-controls { margin: 0.5rem 0 1.5rem; display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; }\n"
           "    .export-controls button { background: rgba(15, 23, 42, 0.85); color: #e2e8f0; border: 1px solid rgba(148, 163, 184, 0.35); box-shadow: 0 16px 36px rgba(2, 6, 23, 0.5); text-transform: none; letter-spacing: 0.04em; padding: 0.75rem 1.5rem; }\n"
           "    .export-controls button:hover { background: rgba(15, 23, 42, 0.92); border-color: rgba(94, 234, 212, 0.55); box-shadow: 0 20px 48px rgba(13, 148, 136, 0.3); }\n"
           "    .export-controls button:focus-visible { box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.35); }\n"
           "    .export-controls button::after { display: none; }\n"
           "    .log { white-space: pre-wrap; background: rgba(15, 23, 42, 0.78); padding: clamp(1.25rem, 3vw, 2rem); border-radius: 24px; border: 1px solid rgba(148, 163, 184, 0.25); box-shadow: 0 24px 60px rgba(2, 6, 23, 0.65); backdrop-filter: blur(16px); }\n"
           "    .message { padding: 1rem 1.25rem; border-radius: 16px; margin-bottom: 0.85rem; background: var(--message-bg, rgba(56, 189, 248, 0.12)); border-left: 4px solid var(--message-border, #38bdf8); box-shadow: 0 12px 28px var(--message-glow, rgba(15, 23, 42, 0.6)); color: #f8fafc; opacity: 0; transform: translateY(16px); transition: transform 0.55s cubic-bezier(0.23, 1, 0.32, 1), opacity 0.55s ease, box-shadow 0.55s ease; }\n"
           "    .message.is-visible { opacity: 1; transform: translateY(0); }\n"
           "    .message.message-summary { background: linear-gradient(135deg, rgba(13, 148, 136, 0.28), rgba(14, 165, 233, 0.3)); border-left-color: #06b6d4; font-style: italic; }\n"
           "    .message.message-summary strong { letter-spacing: 0.12em; color: #bae6fd; }\n"
           "    .message.message-enter { animation: messagePulse 1.2s ease-out forwards; }\n"
           "    @keyframes messagePulse {\n"
           "      0% { box-shadow: 0 0 0 0 var(--message-glow, rgba(56, 189, 248, 0.35)); }\n"
           "      55% { box-shadow: 0 0 36px var(--message-glow, rgba(56, 189, 248, 0.55)); }\n"
           "      100% { box-shadow: 0 12px 28px var(--message-glow, rgba(15, 23, 42, 0.6)); }\n"
           "    }\n"
           "    .message strong { display: block; margin-bottom: 0.35rem; letter-spacing: 0.05em; color: rgba(226, 232, 240, 0.92); }\n"
           "    .research-notes { margin-top: 0.75rem; padding: 0.75rem 1rem; border-radius: 12px; background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(94, 234, 212, 0.35); box-shadow: inset 0 0 0 1px rgba(45, 212, 191, 0.18); }\n"
           "    .research-notes strong { display: block; margin-bottom: 0.5rem; letter-spacing: 0.05em; text-transform: uppercase; color: rgba(94, 234, 212, 0.9); font-size: 0.85rem; }\n"
           "    .research-notes ul { margin: 0; padding-left: 1.15rem; list-style: decimal; color: rgba(226, 232, 240, 0.92); }\n"
           "    .research-notes li { margin-bottom: 0.35rem; }\n"
           "    .research-notes li:last-child { margin-bottom: 0; }\n"
           "    .research-notes a { color: #38bdf8; text-decoration: none; }\n"
           "    .research-notes a:hover { text-decoration: underline; }\n"
           "    .research-notes p { margin: 0 0 0.5rem 0; color: rgba(226, 232, 240, 0.85); }\n"
           "  </style>\n"
           "</head>\n"
           "<body>\n"
           "  <div class=\"card\">\n"
           "    <h1>aiChat Arena</h1>\n"
           "    <p>Configure friendly AI companions, pick their Open WebUI models, and watch them chat about your topic.</p>\n"
           "    <label for=\"topic\">Conversation topic</label>\n"
           "    <input id=\"topic\" placeholder=\"Space exploration strategies\" />\n"
           "    <label for=\"turns\">Number of turns</label>\n"
           "    <input id=\"turns\" type=\"number\" min=\"1\" max=\"12\" value=\"3\" />\n"
           "    <div class=\"search-control\">\n"
           "      <label class=\"form-toggle\">\n"
           "        <input id=\"enableSearch\" type=\"checkbox\" />\n"
           "        <span>Use web search between turns</span>\n"
           "      </label>\n"
           "      <p id=\"searchHelper\" class=\"form-hint\">Enable to gather web research between turns.</p>\n"
           "    </div>\n"
           "    <div class=\"actions\">\n"
           "      <button id=\"addParticipant\">Add participant</button>\n"
           "      <button id=\"start\">Start conversation</button>\n"
           "    </div>\n"
           "    <div id=\"participants\" class=\"participants\"></div>\n"
           "    <div class=\"status-bar\">\n"
           "      <div id=\"connectionFlag\" class=\"connection-flag connection-checking\">\n"
           "        <span class=\"connection-dot\"></span>\n"
           "        <span class=\"connection-text\">Checking API...</span>\n"
           "      </div>\n"
           "      <div id=\"status\"></div>\n"
           "    </div>\n"
           "    <details id=\"connectionDiagnostics\" class=\"connection-diagnostics\">\n"
           "      <summary>API troubleshooting tips</summary>\n"
           "      <div class=\"diagnostics-content\">\n"
           "        <p id=\"diagnosticsSummary\">aiChat will display connection details here after it inspects the Open WebUI API.</p>\n"
           "        <pre><code id=\"diagnosticsCurl\">curl " DEFAULT_WEBUI_BASE "/api/models</code></pre>\n"
           "        <p id=\"diagnosticsNotes\">Run the command above from the machine hosting aiChat to confirm the API responds, then check the terminal for detailed error logs.</p>\n"
           "      </div>\n"
           "    </details>\n"
           "  </div>\n"
           "  <div id=\"transcript\" class=\"log\" style=\"display:none;\">\n"
           "    <h2>Conversation transcript</h2>\n"
           "    <div id=\"exportControls\" class=\"export-controls\" style=\"display:none;\">\n"
           "      <button type=\"button\" id=\"exportText\">Download transcript (text)</button>\n"
           "      <button type=\"button\" id=\"exportJson\">Download transcript (JSON)</button>\n"
           "    </div>\n"
           "    <div id=\"messages\"></div>\n"
           "  </div>\n"
           "  <script>\n"
           "    const participantsEl = document.getElementById('participants');\n"
           "    const statusEl = document.getElementById('status');\n"
           "    const connectionFlagEl = document.getElementById('connectionFlag');\n"
           "    const connectionTextEl = connectionFlagEl ? connectionFlagEl.querySelector('.connection-text') : null;\n"
           "    const diagnosticsDetailsEl = document.getElementById('connectionDiagnostics');\n"
           "    const diagnosticsSummaryEl = document.getElementById('diagnosticsSummary');\n"
           "    const diagnosticsCurlEl = document.getElementById('diagnosticsCurl');\n"
           "    const diagnosticsNotesEl = document.getElementById('diagnosticsNotes');\n"
           "    const enableSearchInput = document.getElementById('enableSearch');\n"
           "    const searchToggleLabel = enableSearchInput ? enableSearchInput.closest('.form-toggle') : null;\n"
           "    const searchHelperEl = document.getElementById('searchHelper');\n"
          "    const searchState = { requested: false, active: false };\n"
           "    const MODEL_LOAD_ERROR_MESSAGE = 'Unable to load models from Open WebUI. Expand the troubleshooting tips below for commands you can try.';\n"
           "    const MODEL_LOAD_ERROR_KEY = 'model-load-error';\n"
           "    const PARTICIPANT_STORAGE_KEY = 'aiChatArena.participants';\n"
           "    let diagnosticsInfo = null;\n"
           "    function loadStoredParticipants() {\n"
           "      if (typeof window === 'undefined' || !window.localStorage) {\n"
           "        return [];\n"
           "      }\n"
           "      try {\n"
           "        const raw = window.localStorage.getItem(PARTICIPANT_STORAGE_KEY);\n"
           "        if (!raw) {\n"
           "          return [];\n"
           "        }\n"
           "        const parsed = JSON.parse(raw);\n"
           "        if (!Array.isArray(parsed)) {\n"
           "          return [];\n"
           "        }\n"
           "        return parsed\n"
           "          .map((entry) => ({\n"
           "            name: entry && typeof entry.name === 'string' ? entry.name.trim() : '',\n"
           "            model: entry && typeof entry.model === 'string' ? entry.model.trim() : '',\n"
           "            displayModel: entry && typeof entry.displayModel === 'string' ? entry.displayModel.trim() : ''\n"
           "          }))\n"
           "          .filter((entry) => entry.name || entry.model || entry.displayModel);\n"
           "      } catch (error) {\n"
           "        return [];\n"
           "      }\n"
           "    }\n"
           "    function persistParticipantsState(participants) {\n"
           "      if (typeof window === 'undefined' || !window.localStorage) {\n"
           "        return;\n"
           "      }\n"
           "      try {\n"
           "        if (!participants || participants.length === 0) {\n"
           "          window.localStorage.removeItem(PARTICIPANT_STORAGE_KEY);\n"
           "          return;\n"
           "        }\n"
           "        window.localStorage.setItem(PARTICIPANT_STORAGE_KEY, JSON.stringify(participants));\n"
           "      } catch (error) {\n"
           "        // Ignore storage errors.\n"
           "      }\n"
           "    }\n"
           "    function snapshotCurrentParticipants() {\n"
           "      if (!participantsEl) {\n"
           "        return [];\n"
           "      }\n"
           "      const nodeList = participantsEl.querySelectorAll('.participant');\n"
           "      const snapshot = [];\n"
           "      Array.from(nodeList).forEach((div) => {\n"
           "        const nameInput = div.querySelector('input[name=\"name\"]');\n"
           "        const selectEl = div.querySelector('select[name=\"model\"]');\n"
           "        const name = nameInput && typeof nameInput.value === 'string' ? nameInput.value.trim() : '';\n"
           "        const rawValue = selectEl && typeof selectEl.value === 'string' ? selectEl.value.trim() : '';\n"
           "        const datasetDesired =\n"
           "          selectEl && selectEl.dataset && typeof selectEl.dataset.desiredModel === 'string'\n"
           "            ? selectEl.dataset.desiredModel.trim()\n"
           "            : '';\n"
           "        const datasetDisplay =\n"
           "          selectEl && selectEl.dataset && typeof selectEl.dataset.displayModel === 'string'\n"
           "            ? selectEl.dataset.displayModel.trim()\n"
           "            : '';\n"
           "        const selectedOption = selectEl ? selectEl.options[selectEl.selectedIndex] : null;\n"
           "        let canonicalValue = '';\n"
           "        if (selectedOption && selectedOption.dataset && typeof selectedOption.dataset.canonicalModel === 'string') {\n"
           "          canonicalValue = selectedOption.dataset.canonicalModel.trim();\n"
           "        }\n"
           "        if (!canonicalValue && rawValue) {\n"
           "          canonicalValue = rawValue;\n"
           "        }\n"
           "        if (!canonicalValue && datasetDesired) {\n"
           "          canonicalValue = datasetDesired;\n"
           "        }\n"
           "        let displayValue = '';\n"
           "        if (selectedOption && selectedOption.dataset && typeof selectedOption.dataset.displayModel === 'string') {\n"
           "          displayValue = selectedOption.dataset.displayModel.trim();\n"
           "        }\n"
           "        if (!displayValue && datasetDisplay) {\n"
           "          displayValue = datasetDisplay;\n"
           "        }\n"
           "        if (!displayValue) {\n"
           "          displayValue = canonicalValue;\n"
           "        }\n"
           "        const entry = { name, model: canonicalValue, displayModel: displayValue };\n"
           "        if (entry.name || entry.model || entry.displayModel) {\n"
           "          snapshot.push(entry);\n"
           "        }\n"
           "      });\n"
           "      return snapshot;\n"
           "    }\n"
           "    function updateStoredParticipants() {\n"
           "      const snapshot = snapshotCurrentParticipants();\n"
           "      persistParticipantsState(snapshot);\n"
           "    }\n"
           "    statusEl.addEventListener('animationend', (event) => {\n"
            "      if (event.animationName === 'statusGlow') {\n"
            "        statusEl.classList.remove('status-flash');\n"
            "      }\n"
            "    });\n"
           "    function setStatus(message, key) {\n"
            "      const text = typeof message === 'string' ? message : (message ? String(message) : '');\n"
            "      const active = Boolean(text);\n"
            "      statusEl.textContent = active ? text : '';\n"
            "      statusEl.classList.toggle('status-active', active);\n"
           "      if (active && typeof key === 'string' && key) {\n"
           "        statusEl.dataset.statusKey = key;\n"
           "      } else if (active) {\n"
           "        delete statusEl.dataset.statusKey;\n"
           "      } else {\n"
           "        delete statusEl.dataset.statusKey;\n"
           "      }\n"
            "      if (active) {\n"
            "        statusEl.classList.remove('status-flash');\n"
            "        void statusEl.offsetWidth;\n"
            "        statusEl.classList.add('status-flash');\n"
            "      } else {\n"
            "        statusEl.classList.remove('status-flash');\n"
            "      }\n"
            "    }\n"
           "    function setConnectionState(state, message) {\n"
           "      if (!connectionFlagEl || !connectionTextEl) {\n"
           "        return;\n"
           "      }\n"
           "      const stateClasses = ['connection-checking', 'connection-connected', 'connection-disconnected'];\n"
           "      stateClasses.forEach((cls) => connectionFlagEl.classList.remove(cls));\n"
           "      let label = 'Checking API...';\n"
           "      if (state === 'connected') {\n"
           "        connectionFlagEl.classList.add('connection-connected');\n"
           "        label = message && typeof message === 'string' && message ? message : 'API connected';\n"
           "      } else if (state === 'disconnected') {\n"
           "        connectionFlagEl.classList.add('connection-disconnected');\n"
           "        label = message && typeof message === 'string' && message ? message : 'API unreachable';\n"
           "      } else {\n"
           "        connectionFlagEl.classList.add('connection-checking');\n"
           "        label = message && typeof message === 'string' && message ? message : 'Checking API...';\n"
           "      }\n"
           "      connectionTextEl.textContent = label;\n"
           "      if (diagnosticsDetailsEl) {\n"
           "        diagnosticsDetailsEl.classList.toggle('diagnostics-error', state === 'disconnected');\n"
           "        if (state === 'disconnected') {\n"
           "          diagnosticsDetailsEl.open = true;\n"
           "        }\n"
           "      }\n"
           "      updateDiagnosticsPanel(state);\n"
           "    }\n"
          "    function updateSearchHelper() {\n"
          "      if (!searchHelperEl) {\n"
          "        return;\n"
          "      }\n"
          "      const requested = Boolean(searchState.requested);\n"
          "      const active = Boolean(searchState.active);\n"
          "      searchHelperEl.classList.remove('is-warning', 'is-active');\n"
          "      if (searchToggleLabel) {\n"
          "        searchToggleLabel.classList.remove('is-disabled');\n"
          "      }\n"
          "      if (enableSearchInput) {\n"
          "        enableSearchInput.removeAttribute('aria-disabled');\n"
          "      }\n"
          "      if (active) {\n"
          "        searchHelperEl.textContent = 'Web search is active for this conversation.';\n"
          "        searchHelperEl.classList.add('is-active');\n"
          "        return;\n"
          "      }\n"
          "      if (requested) {\n"
          "        searchHelperEl.textContent = 'Web search will run during the next conversation.';\n"
          "        searchHelperEl.classList.add('is-active');\n"
          "        return;\n"
          "      }\n"
          "      searchHelperEl.textContent = 'Enable to gather web research between turns.';\n"
          "    }\n"
           "    if (enableSearchInput) {\n"
           "      searchState.requested = enableSearchInput.checked;\n"
           "      enableSearchInput.addEventListener('change', () => {\n"
           "        searchState.requested = enableSearchInput.checked;\n"
           "        updateSearchHelper();\n"
           "      });\n"
           "    }\n"
           "    updateSearchHelper();\n"
           "    setConnectionState('checking');\n"
           "    function updateDiagnosticsPanel(state) {\n"
           "      if (!diagnosticsDetailsEl) {\n"
           "        return;\n"
           "      }\n"
          "      const info = diagnosticsInfo || {};\n"
          "      const modelsUrl = typeof info.modelsUrl === 'string' ? info.modelsUrl.trim() : '';\n"
          "      const fallbackUrl = typeof info.fallbackUrl === 'string' ? info.fallbackUrl.trim() : '';\n"
          "      const endpoint = typeof info.webuiEndpoint === 'string' ? info.webuiEndpoint.trim() : '';\n"
          "      const usesApiKey = Boolean(info.usesApiKey);\n"
          "      const preferredUrl = modelsUrl || fallbackUrl;\n"
          "      if (!searchState.active && enableSearchInput) {\n"
          "        searchState.requested = enableSearchInput.checked;\n"
          "      }\n"
          "      updateSearchHelper();\n"
           "      if (diagnosticsSummaryEl) {\n"
           "        const parts = [];\n"
           "        if (modelsUrl) {\n"
           "          parts.push(`Primary models endpoint: ${modelsUrl}.`);\n"
           "        } else {\n"
           "          parts.push('Primary models endpoint could not be derived from the configured URL.');\n"
           "        }\n"
           "        if (fallbackUrl && fallbackUrl !== modelsUrl) {\n"
           "          parts.push(`Legacy fallback endpoint: ${fallbackUrl}.`);\n"
           "        }\n"
           "        if (endpoint) {\n"
           "          parts.push(`Conversations will be sent to: ${endpoint}.`);\n"
           "        }\n"
           "        diagnosticsSummaryEl.textContent = parts.join(' ');\n"
           "      }\n"
           "      if (diagnosticsCurlEl) {\n"
           "        const commandParts = ['curl'];\n"
           "        if (usesApiKey) {\n"
           "          commandParts.push('-H \"Authorization: Bearer <YOUR_WEBUI_API_KEY>\"');\n"
           "        }\n"
           "        if (preferredUrl) {\n"
           "          commandParts.push(`\"${preferredUrl}\"`);\n"
           "        } else {\n"
           "          commandParts.push('\"" DEFAULT_WEBUI_BASE "/api/models\"');\n"
           "        }\n"
           "        if (state === 'disconnected') {\n"
           "          commandParts.push('--verbose');\n"
           "        }\n"
           "        diagnosticsCurlEl.textContent = commandParts.join(' ');\n"
           "      }\n"
           "      if (diagnosticsNotesEl) {\n"
           "        const targetHint = preferredUrl || endpoint || 'your Open WebUI host';\n"
           "        let notesText = `Run the command above from the machine hosting aiChat to confirm ${targetHint} is reachable. If it fails, adjust the OLLAMA_URL (currently ${endpoint || 'not set'}) or check your firewall.`;\n"
           "        diagnosticsNotesEl.textContent = notesText;\n"
           "      }\n"
           "    }\n"
           "    async function loadDiagnosticsInfo() {\n"
           "      try {\n"
           "        const response = await fetch('/diagnostics');\n"
           "        if (!response.ok) {\n"
           "          return;\n"
           "        }\n"
           "        diagnosticsInfo = await response.json();\n"
           "        updateDiagnosticsPanel();\n"
           "      } catch (error) {\n"
           "        // Ignore diagnostics fetch errors; the panel will keep its placeholder text.\n"
           "      }\n"
           "    }\n"
           "    updateDiagnosticsPanel();\n"
           "    loadDiagnosticsInfo();\n"
           "    const messagesEl = document.getElementById('messages');\n"
           "    const transcriptEl = document.getElementById('transcript');\n"
           "    const exportControlsEl = document.getElementById('exportControls');\n"
           "    const exportTextButton = document.getElementById('exportText');\n"
           "    const exportJsonButton = document.getElementById('exportJson');\n"
           "    if (exportTextButton) {\n"
           "      exportTextButton.addEventListener('click', (event) => {\n"
           "        event.preventDefault();\n"
           "        exportTranscriptAsText();\n"
           "      });\n"
           "    }\n"
           "    if (exportJsonButton) {\n"
           "      exportJsonButton.addEventListener('click', (event) => {\n"
           "        event.preventDefault();\n"
           "        exportTranscriptAsJson();\n"
           "      });\n"
           "    }\n"
           "    const transcriptMessages = [];\n"
           "    const transcriptEntries = [];\n"
           "    const TEXT_NEWLINE = String.fromCharCode(10);\n"
           "    const DOUBLE_NEWLINE = TEXT_NEWLINE + TEXT_NEWLINE;\n"
           "    let summaryAppended = false;\n"
           "    let conversationComplete = false;\n"
           "    let conversationMetadata = { searchRequested: false, searchEnabled: false, completedAt: null };\n"
           "    let currentTopic = '';\n"
           "    let currentTurns = 0;\n"
           "    let currentParticipants = [];\n"
           "    let availableModels = [];\n"
           "    const modelSelects = new Set();\n"
           "    let modelLoadError = false;\n"
           "    let missingModelWarning = false;\n"
           "    const basePalette = [\n"
           "      {\n"
           "        messageBackground: 'linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(59, 130, 246, 0.45))',\n"
           "        border: '#38bdf8',\n"
           "        glow: 'rgba(56, 189, 248, 0.45)',\n"
           "        cardBackground: 'linear-gradient(160deg, rgba(12, 74, 110, 0.85), rgba(37, 99, 235, 0.75))',\n"
           "        cardPattern: 'radial-gradient(circle at 20% 20%, rgba(14, 165, 233, 0.35) 0, transparent 45%), radial-gradient(circle at 80% 0%, rgba(59, 130, 246, 0.3) 0, transparent 40%)'\n"
           "      },\n"
           "      {\n"
           "        messageBackground: 'linear-gradient(135deg, rgba(244, 114, 182, 0.2), rgba(236, 72, 153, 0.45))',\n"
           "        border: '#f472b6',\n"
           "        glow: 'rgba(244, 114, 182, 0.45)',\n"
           "        cardBackground: 'linear-gradient(160deg, rgba(88, 28, 135, 0.82), rgba(162, 28, 175, 0.78))',\n"
           "        cardPattern: 'radial-gradient(circle at 25% 20%, rgba(249, 168, 212, 0.32) 0, transparent 42%), radial-gradient(circle at 80% 10%, rgba(236, 72, 153, 0.28) 0, transparent 38%)'\n"
           "      },\n"
           "      {\n"
           "        messageBackground: 'linear-gradient(135deg, rgba(52, 211, 153, 0.2), rgba(16, 185, 129, 0.45))',\n"
           "        border: '#34d399',\n"
           "        glow: 'rgba(16, 185, 129, 0.4)',\n"
           "        cardBackground: 'linear-gradient(160deg, rgba(4, 47, 46, 0.85), rgba(13, 148, 136, 0.78))',\n"
           "        cardPattern: 'radial-gradient(circle at 18% 22%, rgba(94, 234, 212, 0.32) 0, transparent 45%), radial-gradient(circle at 82% 12%, rgba(16, 185, 129, 0.3) 0, transparent 40%)'\n"
           "      },\n"
           "      {\n"
           "        messageBackground: 'linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(249, 115, 22, 0.45))',\n"
           "        border: '#f59e0b',\n"
           "        glow: 'rgba(251, 146, 60, 0.45)',\n"
           "        cardBackground: 'linear-gradient(160deg, rgba(88, 40, 12, 0.85), rgba(234, 88, 12, 0.75))',\n"
           "        cardPattern: 'radial-gradient(circle at 24% 16%, rgba(251, 191, 36, 0.32) 0, transparent 44%), radial-gradient(circle at 78% 10%, rgba(249, 115, 22, 0.28) 0, transparent 40%)'\n"
           "      },\n"
           "      {\n"
           "        messageBackground: 'linear-gradient(135deg, rgba(129, 140, 248, 0.2), rgba(99, 102, 241, 0.45))',\n"
           "        border: '#818cf8',\n"
           "        glow: 'rgba(129, 140, 248, 0.45)',\n"
           "        cardBackground: 'linear-gradient(160deg, rgba(30, 41, 102, 0.85), rgba(76, 29, 149, 0.78))',\n"
           "        cardPattern: 'radial-gradient(circle at 22% 24%, rgba(129, 140, 248, 0.32) 0, transparent 46%), radial-gradient(circle at 84% 14%, rgba(165, 180, 252, 0.28) 0, transparent 40%)'\n"
           "      },\n"
           "      {\n"
           "        messageBackground: 'linear-gradient(135deg, rgba(45, 212, 191, 0.2), rgba(59, 130, 246, 0.42))',\n"
           "        border: '#5eead4',\n"
           "        glow: 'rgba(56, 189, 248, 0.38)',\n"
           "        cardBackground: 'linear-gradient(160deg, rgba(8, 47, 73, 0.85), rgba(30, 64, 175, 0.78))',\n"
           "        cardPattern: 'radial-gradient(circle at 28% 18%, rgba(56, 189, 248, 0.32) 0, transparent 42%), radial-gradient(circle at 76% 8%, rgba(45, 212, 191, 0.28) 0, transparent 38%)'\n"
           "      }\n"
           "    ];\n"
           "    const participantStyles = new Map();\n"
           "    function computePaletteEntry(index) {\n"
           "      const position = Number.isFinite(index) && index >= 0 ? index : 0;\n"
           "      if (position < basePalette.length) {\n"
           "        return { ...basePalette[position] };\n"
           "      }\n"
           "      const hue = (position * 47) % 360;\n"
           "      const accent = (hue + 60) % 360;\n"
           "      return {\n"
           "        messageBackground: `linear-gradient(135deg, hsla(${hue}, 80%, 22%, 0.55), hsla(${accent}, 85%, 36%, 0.75))`,\n"
           "        border: `hsla(${accent}, 90%, 65%, 1)`,\n"
           "        glow: `hsla(${accent}, 90%, 70%, 0.45)`,\n"
           "        cardBackground: `linear-gradient(160deg, hsla(${hue}, 65%, 16%, 0.9), hsla(${accent}, 70%, 22%, 0.95))`,\n"
           "        cardPattern: `radial-gradient(circle at 20% 25%, hsla(${accent}, 85%, 60%, 0.35) 0, transparent 42%), radial-gradient(circle at 80% 10%, hsla(${hue}, 85%, 55%, 0.3) 0, transparent 38%)`\n"
           "      };\n"
           "    }\n"
           "    function getPaletteForIndex(index) {\n"
           "      const palette = computePaletteEntry(index);\n"
           "      if (!palette.glow) {\n"
           "        palette.glow = 'rgba(59, 130, 246, 0.35)';\n"
           "      }\n"
           "      if (!palette.messageBackground) {\n"
           "        palette.messageBackground = 'linear-gradient(135deg, rgba(59, 130, 246, 0.18), rgba(129, 140, 248, 0.4))';\n"
           "      }\n"
           "      if (!palette.cardBackground) {\n"
           "        palette.cardBackground = 'linear-gradient(160deg, rgba(15, 23, 42, 0.85), rgba(30, 64, 175, 0.78))';\n"
           "      }\n"
           "      return palette;\n"
           "    }\n"
           "    function assignParticipantStyles(participants) {\n"
           "      participantStyles.clear();\n"
           "      participants.forEach((participant, index) => {\n"
           "        participantStyles.set(index, getPaletteForIndex(index));\n"
           "      });\n"
           "    }\n"
           "    function normaliseParticipants(list) {\n"
           "      if (!Array.isArray(list)) {\n"
           "        return [];\n"
           "      }\n"
           "      return list\n"
           "        .map((participant) => ({\n"
           "          name: participant && typeof participant.name === 'string' ? participant.name.trim() : '',\n"
           "          model: participant && typeof participant.model === 'string' ? participant.model.trim() : '',\n"
           "          displayModel: participant && typeof participant.displayModel === 'string' ? participant.displayModel.trim() : ''\n"
           "        }))\n"
           "        .filter((participant) => participant.name || participant.model || participant.displayModel);\n"
           "    }\n"
           "    function resetTranscriptState(topic, turns, participants) {\n"
           "      transcriptMessages.length = 0;\n"
           "      transcriptEntries.length = 0;\n"
           "      summaryAppended = false;\n"
           "      conversationComplete = false;\n"
           "      currentTopic = typeof topic === 'string' ? topic : '';\n"
           "      currentTurns = Number.isFinite(turns) ? turns : 0;\n"
           "      currentParticipants = normaliseParticipants(participants);\n"
           "      conversationMetadata = {\n"
           "        searchRequested: enableSearchInput ? !!enableSearchInput.checked : false,\n"
           "        searchEnabled: false,\n"
           "        completedAt: null\n"
           "      };\n"
           "      updateExportControls();\n"
           "    }\n"
           "    function cloneResearchData(research) {\n"
           "      if (!research || typeof research !== 'object') {\n"
           "        return null;\n"
           "      }\n"
           "      try {\n"
           "        return JSON.parse(JSON.stringify(research));\n"
           "      } catch (error) {\n"
           "        return null;\n"
           "      }\n"
           "    }\n"
           "    function updateExportControls() {\n"
           "      if (!exportControlsEl) {\n"
           "        return;\n"
           "      }\n"
           "      if (!conversationComplete || transcriptEntries.length === 0) {\n"
           "        exportControlsEl.style.display = 'none';\n"
           "        return;\n"
           "      }\n"
           "      exportControlsEl.style.display = 'flex';\n"
           "    }\n"
           "    function markTranscriptComplete() {\n"
           "      if (!conversationComplete) {\n"
           "        conversationComplete = true;\n"
           "      }\n"
           "      if (!conversationMetadata.completedAt) {\n"
           "        conversationMetadata.completedAt = new Date().toISOString();\n"
           "      }\n"
           "      updateExportControls();\n"
           "    }\n"
           "    function formatResearchNotesForText(research) {\n"
           "      if (!research || typeof research !== 'object') {\n"
           "        return [];\n"
           "      }\n"
           "      const lines = [];\n"
           "      const query = typeof research.query === 'string' ? research.query.trim() : '';\n"
           "      const answer = typeof research.answer === 'string' ? research.answer.trim() : '';\n"
           "      if (query) {\n"
           "        lines.push(`  Query: ${query}`);\n"
           "      }\n"
           "      if (answer) {\n"
           "        lines.push(`  Summary: ${answer}`);\n"
           "      }\n"
           "      const results = Array.isArray(research.results) ? research.results : [];\n"
           "      results.slice(0, 3).forEach((item, index) => {\n"
           "        if (!item || typeof item !== 'object') {\n"
           "          return;\n"
           "        }\n"
           "        const title = typeof item.title === 'string' ? item.title.trim() : '';\n"
           "        const snippet = typeof item.snippet === 'string' ? item.snippet.trim() : '';\n"
           "        const url = typeof item.url === 'string' ? item.url.trim() : '';\n"
           "        const pieces = [];\n"
           "        if (title) {\n"
           "          pieces.push(title);\n"
           "        }\n"
           "        if (snippet) {\n"
           "          pieces.push(snippet);\n"
           "        }\n"
           "        if (url) {\n"
           "          pieces.push(url);\n"
           "        }\n"
           "        if (pieces.length) {\n"
           "          lines.push(`  ${index + 1}. ${pieces.join(' — ')}`);\n"
           "        }\n"
           "      });\n"
           "      return lines;\n"
           "    }\n"
           "    function normaliseTopicSlug(value) {\n"
           "      if (!value || typeof value !== 'string') {\n"
           "        return 'conversation';\n"
           "      }\n"
           "      const trimmed = value.trim().toLowerCase();\n"
           "      if (!trimmed) {\n"
           "        return 'conversation';\n"
           "      }\n"
           "      const slug = trimmed.replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60);\n"
           "      return slug || 'conversation';\n"
           "    }\n"
           "    function resolveCompletionTimestamp() {\n"
           "      if (conversationMetadata && conversationMetadata.completedAt) {\n"
           "        const parsed = new Date(conversationMetadata.completedAt);\n"
           "        if (!Number.isNaN(parsed.getTime())) {\n"
           "          return parsed;\n"
           "        }\n"
           "      }\n"
           "      return new Date();\n"
           "    }\n"
           "    function buildTranscriptFilename(extension) {\n"
           "      const topicSlug = normaliseTopicSlug(typeof currentTopic === 'string' ? currentTopic : '');\n"
           "      const timestamp = resolveCompletionTimestamp().toISOString().replace(/[:.]/g, '-');\n"
           "      return `${topicSlug}-${timestamp}.${extension}`;\n"
           "    }\n"
           "    function buildTranscriptText() {\n"
           "      if (transcriptEntries.length === 0) {\n"
           "        return '';\n"
           "      }\n"
           "      const topic = typeof currentTopic === 'string' ? currentTopic.trim() : '';\n"
           "      const participantsList = Array.isArray(currentParticipants) ? currentParticipants : [];\n"
           "      const participantNames = participantsList\n"
           "        .map((entry) => (entry && typeof entry.name === 'string' ? entry.name.trim() : ''))\n"
           "        .filter((name) => name);\n"
           "      const turnsValue = Number.isFinite(currentTurns) && currentTurns > 0 ? currentTurns : transcriptMessages.length;\n"
           "      const headerLines = ['aiChat Arena transcript'];\n"
           "      if (topic) {\n"
           "        headerLines.push(`Topic: ${topic}`);\n"
           "      }\n"
           "      if (participantNames.length) {\n"
           "        headerLines.push(`Participants: ${participantNames.join(', ')}`);\n"
           "      }\n"
           "      if (Number.isFinite(turnsValue) && turnsValue > 0) {\n"
           "        headerLines.push(`Turns: ${turnsValue}`);\n"
           "      }\n"
           "      if (conversationMetadata) {\n"
           "        if (conversationMetadata.searchEnabled) {\n"
           "          headerLines.push('Web search: enabled');\n"
           "        } else if (conversationMetadata.searchRequested) {\n"
           "          headerLines.push('Web search: requested (not available)');\n"
           "        } else {\n"
           "          headerLines.push('Web search: disabled');\n"
           "        }\n"
           "        if (conversationMetadata.completedAt) {\n"
           "          headerLines.push(`Completed: ${conversationMetadata.completedAt}`);\n"
           "        }\n"
           "      }\n"
           "      headerLines.push(`Exported: ${new Date().toISOString()}`);\n"
           "      const blocks = transcriptEntries.map((entry) => {\n"
           "        const lines = [];\n"
           "        const label = entry.type === 'summary'\n"
           "          ? 'Summary'\n"
           "          : (entry.name || (Number.isInteger(entry.participantIndex) ? `Participant ${entry.participantIndex + 1}` : 'Message'));\n"
           "        lines.push(`${label}:`);\n"
           "        if (entry.plainText) {\n"
           "          lines.push(entry.plainText);\n"
           "        }\n"
           "        const researchLines = formatResearchNotesForText(entry.research);\n"
           "        if (researchLines.length) {\n"
           "          lines.push('Research notes:');\n"
           "          researchLines.forEach((line) => lines.push(line));\n"
           "        }\n"
           "        return lines.join(TEXT_NEWLINE);\n"
           "      });\n"
           "      return `${headerLines.join(TEXT_NEWLINE)}${DOUBLE_NEWLINE}${blocks.join(DOUBLE_NEWLINE)}`.trim();\n"
           "    }\n"
           "    function buildTranscriptJson() {\n"
           "      const participantsList = Array.isArray(currentParticipants) ? currentParticipants : [];\n"
           "      const participants = participantsList.map((participant) => ({\n"
           "        name: participant && typeof participant.name === 'string' ? participant.name : '',\n"
           "        model: participant && typeof participant.model === 'string' ? participant.model : '',\n"
           "        displayModel: participant && typeof participant.displayModel === 'string' ? participant.displayModel : ''\n"
           "      }));\n"
           "      const turnsValue = Number.isFinite(currentTurns) && currentTurns > 0 ? currentTurns : transcriptMessages.length;\n"
           "      const messages = transcriptEntries.map((entry) => {\n"
           "        const result = {\n"
           "          type: entry.type,\n"
           "          name: entry.name,\n"
           "          model: entry.model,\n"
           "          displayModel: entry.displayModel,\n"
           "          turn: entry.turn,\n"
           "          participantIndex: entry.participantIndex,\n"
           "          html: entry.html,\n"
           "          plainText: entry.plainText\n"
           "        };\n"
           "        if (entry.research) {\n"
           "          result.research = entry.research;\n"
           "        }\n"
           "        return result;\n"
           "      });\n"
           "      return {\n"
           "        topic: typeof currentTopic === 'string' ? currentTopic : '',\n"
           "        turns: turnsValue,\n"
           "        participants,\n"
           "        metadata: {\n"
           "          searchRequested: Boolean(conversationMetadata && conversationMetadata.searchRequested),\n"
           "          searchEnabled: Boolean(conversationMetadata && conversationMetadata.searchEnabled),\n"
           "          completedAt: conversationMetadata && conversationMetadata.completedAt ? conversationMetadata.completedAt : null,\n"
           "          exportedAt: new Date().toISOString()\n"
           "        },\n"
           "        messages\n"
           "      };\n"
           "    }\n"
           "    function downloadBlob(content, mimeType, filename) {\n"
           "      const blob = new Blob([content], { type: mimeType });\n"
           "      const url = URL.createObjectURL(blob);\n"
           "      const link = document.createElement('a');\n"
           "      link.href = url;\n"
           "      link.download = filename;\n"
           "      document.body.appendChild(link);\n"
           "      link.click();\n"
           "      setTimeout(() => {\n"
           "        document.body.removeChild(link);\n"
           "        URL.revokeObjectURL(url);\n"
           "      }, 0);\n"
           "    }\n"
           "    function exportTranscriptAsText() {\n"
           "      if (transcriptEntries.length === 0) {\n"
           "        return;\n"
           "      }\n"
           "      const payload = buildTranscriptText();\n"
           "      if (!payload) {\n"
           "        return;\n"
           "      }\n"
           "      downloadBlob(payload, 'text/plain;charset=utf-8', buildTranscriptFilename('txt'));\n"
           "    }\n"
           "    function exportTranscriptAsJson() {\n"
           "      if (transcriptEntries.length === 0) {\n"
           "        return;\n"
           "      }\n"
           "      const payload = buildTranscriptJson();\n"
           "      downloadBlob(JSON.stringify(payload, null, 2), 'application/json;charset=utf-8', buildTranscriptFilename('json'));\n"
           "    }\n"
           "    function recordTranscriptMessage(message) {\n"
           "      if (!message) {\n"
           "        return;\n"
           "      }\n"
           "      const entry = {\n"
           "        type: message.isSummary ? 'summary' : 'message',\n"
           "        turn: Number.isFinite(message.turn) ? message.turn : null,\n"
           "        participantIndex: Number.isFinite(message.participantIndex) ? message.participantIndex : null,\n"
           "        name: typeof message.name === 'string' ? message.name : '',\n"
           "        model: typeof message.model === 'string' ? message.model : '',\n"
           "        displayModel: typeof message.displayModel === 'string' ? message.displayModel : '',\n"
           "        html: typeof message.text === 'string' ? message.text : '',\n"
           "        plainText: extractPlainText(message.text),\n"
           "        research: cloneResearchData(message.research)\n"
           "      };\n"
           "      transcriptEntries.push(entry);\n"
           "      if (message.isSummary) {\n"
           "        return;\n"
           "      }\n"
           "      transcriptMessages.push({ name: entry.name, text: entry.html });\n"
           "    }\n"
          "    function extractPlainText(html) {\n"
          "      if (!html) {\n"
          "        return '';\n"
          "      }\n"
          "      const temp = document.createElement('div');\n"
          "      temp.innerHTML = html;\n"
          "      const text = temp.textContent || temp.innerText || '';\n"
          "      return text.replace(/\\s+/g, ' ').trim();\n"
          "    }\n"
          "    function selectSentenceSnippet(html, maxLength) {\n"
          "      const plain = extractPlainText(html);\n"
          "      if (!plain) {\n"
          "        return '';\n"
          "      }\n"
          "      const sentenceMatch = plain.match(/[^.!?]+[.!?]?/);\n"
          "      let sentence = sentenceMatch ? sentenceMatch[0].trim() : plain.trim();\n"
          "      const limit = Number.isFinite(maxLength) && maxLength > 0 ? maxLength : 160;\n"
          "      if (sentence.length > limit) {\n"
          "        sentence = `${sentence.slice(0, limit - 3).replace(/\\s+$/g, '')}…`;\n"
          "      }\n"
          "      return sentence;\n"
          "    }\n"
          "    function formatParticipantSubject(names) {\n"
          "      if (!Array.isArray(names) || names.length === 0) {\n"
          "        return 'The participants';\n"
          "      }\n"
          "      if (names.length === 1) {\n"
          "        return names[0];\n"
          "      }\n"
          "      if (names.length === 2) {\n"
          "        return `${names[0]} and ${names[1]}`;\n"
          "      }\n"
          "      const allButLast = names.slice(0, -1).join(', ');\n"
          "      const last = names[names.length - 1];\n"
          "      return `${allButLast}, and ${last}`;\n"
          "    }\n"
          "    function resolveTopicSummary() {\n"
          "      const trimmedTopic = typeof currentTopic === 'string' ? currentTopic.trim() : '';\n"
          "      const isoDateTime = /^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?Z$/i;\n"
          "      const isoDateOnly = /^\\d{4}-\\d{2}-\\d{2}$/;\n"
          "      if (trimmedTopic && !isoDateTime.test(trimmedTopic) && !isoDateOnly.test(trimmedTopic)) {\n"
          "        return { text: trimmedTopic, derived: false };\n"
          "      }\n"
          "      for (const entry of transcriptMessages) {\n"
          "        const snippet = selectSentenceSnippet(entry && entry.text, 120);\n"
          "        if (snippet) {\n"
          "          return { text: snippet, derived: true };\n"
          "        }\n"
          "      }\n"
          "      return { text: 'their discussion', derived: true };\n"
          "    }\n"
          "    function summariseDiscussionIfNeeded() {\n"
           "      if (summaryAppended || transcriptMessages.length === 0) {\n"
           "        return;\n"
           "      }\n"
           "      const participantsList = Array.isArray(currentParticipants) ? currentParticipants : [];\n"
          "      const participantNames = participantsList\n"
          "        .map((participant) => (participant && participant.name ? participant.name : ''))\n"
          "        .filter((name) => name);\n"
          "      const subject = formatParticipantSubject(participantNames);\n"
          "      const topicSummary = resolveTopicSummary();\n"
           "      const turnsValue = Number.isFinite(currentTurns) && currentTurns > 0 ? currentTurns : transcriptMessages.length;\n"
           "      const turnsText = turnsValue === 1 ? '1 turn' : `${turnsValue} turns`;\n"
           "      const latestByParticipant = new Map();\n"
           "      transcriptMessages.forEach((entry) => {\n"
           "        if (entry.name) {\n"
           "          latestByParticipant.set(entry.name, entry.text || '');\n"
           "        }\n"
           "      });\n"
           "      const highlightSnippets = [];\n"
          "      latestByParticipant.forEach((text, name) => {\n"
          "        const sentence = selectSentenceSnippet(text, 160);\n"
          "        if (sentence) {\n"
          "          highlightSnippets.push(`${name}: ${sentence}`);\n"
          "        }\n"
          "      });\n"
          "      const highlightsText = highlightSnippets.length ? ` Highlights — ${highlightSnippets.join(' | ')}` : '';\n"
          "      let summaryBody = '';\n"
          "      if (!topicSummary.text) {\n"
          "        summaryBody = `${subject} chatted over ${turnsText}.`;\n"
          "      } else if (topicSummary.text === 'their discussion') {\n"
          "        summaryBody = `${subject} shared their thoughts over ${turnsText}.`;\n"
          "      } else if (topicSummary.derived) {\n"
          "        summaryBody = `${subject} explored ${topicSummary.text} over ${turnsText}.`;\n"
          "      } else {\n"
          "        summaryBody = `${subject} discussed \"${topicSummary.text}\" over ${turnsText}.`;\n"
          "      }\n"
          "      summaryBody += highlightsText;\n"
           "      appendMessage({\n"
           "        participantIndex: -1,\n"
           "        name: 'Summary',\n"
           "        model: '',\n"
           "        displayModel: '',\n"
           "        text: `<p>${summaryBody}</p>`,\n"
           "        isSummary: true\n"
           "      });\n"
           "      summaryAppended = true;\n"
           "      markTranscriptComplete();\n"
           "    }\n"
           "    function updateParticipantCardThemes() {\n"
          "      const cards = Array.from(participantsEl.querySelectorAll('.participant'));\n"
          "      cards.forEach((card, index) => {\n"
          "        const palette = getPaletteForIndex(index);\n"
          "        card.style.setProperty('--participant-border', palette.border);\n"
          "        card.style.setProperty('--participant-pattern', palette.cardPattern || 'rgba(59, 130, 246, 0.25)');\n"
          "        card.style.setProperty('--participant-surface', palette.cardBackground);\n"
          "        card.style.setProperty('--participant-glow', palette.glow || 'rgba(59, 130, 246, 0.35)');\n"
          "        card.style.setProperty('--participant-highlight', palette.border || 'rgba(148, 163, 184, 0.8)');\n"
          "      });\n"
          "    }\n"
           "    function renderResearchNotes(research) {\n"
           "      if (!research || typeof research !== 'object') {\n"
           "        return '';\n"
           "      }\n"
           "      const results = Array.isArray(research.results) ? research.results : [];\n"
           "      const answer = typeof research.answer === 'string' ? research.answer.trim() : '';\n"
           "      const query = typeof research.query === 'string' ? research.query.trim() : '';\n"
           "      if (!results.length && !answer) {\n"
           "        return '';\n"
           "      }\n"
           "      const wrapper = document.createElement('div');\n"
           "      wrapper.className = 'research-notes';\n"
           "      const heading = document.createElement('strong');\n"
           "      heading.textContent = query ? `Research notes for \"${query}\"` : 'Research notes';\n"
           "      wrapper.appendChild(heading);\n"
           "      if (answer) {\n"
           "        const summary = document.createElement('p');\n"
           "        summary.textContent = `Summary: ${answer}`;\n"
           "        wrapper.appendChild(summary);\n"
           "      }\n"
           "      if (results.length) {\n"
           "        const list = document.createElement('ul');\n"
           "        results.slice(0, 3).forEach((entry) => {\n"
           "          if (!entry || typeof entry !== 'object') {\n"
           "            return;\n"
           "          }\n"
           "          const title = typeof entry.title === 'string' ? entry.title.trim() : '';\n"
           "          const snippet = typeof entry.snippet === 'string' ? entry.snippet.trim() : '';\n"
           "          const url = typeof entry.url === 'string' ? entry.url.trim() : '';\n"
           "          if (!title && !snippet && !url) {\n"
           "            return;\n"
           "          }\n"
           "          const item = document.createElement('li');\n"
           "          if (title) {\n"
           "            if (url) {\n"
           "              const link = document.createElement('a');\n"
           "              link.href = url;\n"
           "              link.target = '_blank';\n"
           "              link.rel = 'noreferrer noopener';\n"
           "              link.textContent = title;\n"
           "              item.appendChild(link);\n"
           "            } else {\n"
           "              item.appendChild(document.createTextNode(title));\n"
           "            }\n"
           "          } else if (url) {\n"
           "            const link = document.createElement('a');\n"
           "            link.href = url;\n"
           "            link.target = '_blank';\n"
           "            link.rel = 'noreferrer noopener';\n"
           "            link.textContent = url;\n"
           "            item.appendChild(link);\n"
           "          }\n"
           "          if (snippet) {\n"
           "            if (item.childNodes.length) {\n"
           "              item.appendChild(document.createTextNode(' — '));\n"
           "            }\n"
           "            item.appendChild(document.createTextNode(snippet));\n"
           "          }\n"
           "          list.appendChild(item);\n"
           "        });\n"
           "        if (list.children.length) {\n"
           "          wrapper.appendChild(list);\n"
           "        }\n"
           "      }\n"
           "      const temp = document.createElement('div');\n"
           "      temp.appendChild(wrapper);\n"
           "      return temp.innerHTML;\n"
           "    }\n"
           "    function appendMessage(message) {\n"
            "      if (!message || typeof message !== 'object') {\n"
            "        return;\n"
            "      }\n"
            "      const item = document.createElement('div');\n"
           "      item.className = 'message message-enter';\n"
           "      if (message.isSummary) {\n"
           "        item.classList.add('message-summary');\n"
           "      }\n"
           "      const paletteIndex = Number.isFinite(message.participantIndex) && message.participantIndex >= 0\n"
           "        ? message.participantIndex\n"
           "        : 0;\n"
           "      const paletteEntry = participantStyles.get(paletteIndex);\n"
           "      const palette = paletteEntry || getPaletteForIndex(paletteIndex);\n"
           "      item.style.setProperty('--message-bg', palette.messageBackground);\n"
           "      item.style.setProperty('--message-border', palette.border);\n"
           "      item.style.setProperty('--message-glow', palette.glow || 'rgba(59, 130, 246, 0.35)');\n"
          "      const displayModel = (message.displayModel && typeof message.displayModel === 'string') ? message.displayModel : '';\n"
           "      const canonicalModel = (message.model && typeof message.model === 'string') ? message.model : '';\n"
           "      let modelLabel = displayModel;\n"
           "      if (displayModel && canonicalModel && displayModel !== canonicalModel) {\n"
           "        modelLabel = `${displayModel} (${canonicalModel})`;\n"
           "      } else if (!modelLabel && canonicalModel) {\n"
           "        modelLabel = canonicalModel;\n"
           "      }\n"
           "      const header = modelLabel\n"
           "        ? `<strong>${message.name} <span style=\"color:#94a3b8; font-weight:400;\">(${modelLabel})</span></strong>`\n"
           "        : `<strong>${message.name}</strong>`;\n"
           "      const bodyText = typeof message.text === 'string' ? message.text : '';\n"
           "      const researchHtml = renderResearchNotes(message.research);\n"
           "      item.innerHTML = `${header}${bodyText}${researchHtml}`;\n"
          "      messagesEl.appendChild(item);\n"
          "      recordTranscriptMessage(message);\n"
          "      item.addEventListener('animationend', (event) => {\n"
          "        if (event.animationName === 'messagePulse') {\n"
          "          item.classList.remove('message-enter');\n"
          "        }\n"
          "      });\n"
          "      requestAnimationFrame(() => {\n"
          "        requestAnimationFrame(() => {\n"
          "          item.classList.add('is-visible');\n"
          "        });\n"
          "      });\n"
          "      transcriptEl.style.display = 'block';\n"
          "      transcriptEl.scrollTop = transcriptEl.scrollHeight;\n"
          "    }\n"
          "    function populateModelOptions(select, selectedModel) {\n"
          "      const datasetValue = (select.dataset.desiredModel || '').trim();\n"
          "      const datasetDisplay = (select.dataset.displayModel || '').trim();\n"
          "      const providedValue = (selectedModel && typeof selectedModel === 'string') ? selectedModel.trim() : '';\n"
          "      const currentValue = (select.value && typeof select.value === 'string') ? select.value.trim() : '';\n"
          "      const requestedValue = providedValue || datasetValue || currentValue;\n"
          "      const requestedDisplay = datasetDisplay || requestedValue;\n"
          "      const suggestedValue = (select.dataset.suggestedModel || '').trim();\n"
          "      select.innerHTML = '';\n"
          "      const placeholder = document.createElement('option');\n"
          "      placeholder.value = '';\n"
          "      placeholder.textContent = modelLoadError\n"
           "        ? 'Unable to load models'\n"
           "        : (availableModels.length ? 'Select a model' : 'Loading models...');\n"
           "      placeholder.disabled = availableModels.length > 0;\n"
           "      select.appendChild(placeholder);\n"
           "      let hasMatch = false;\n"
           "      let matchedDisplay = '';\n"
           "      availableModels.forEach((item) => {\n"
           "        const option = document.createElement('option');\n"
           "        const canonical = item && typeof item.model === 'string' ? item.model : '';\n"
           "        const alias = item && typeof item.name === 'string' && item.name ? item.name : canonical;\n"
           "        option.value = canonical;\n"
           "        option.dataset.canonicalModel = canonical;\n"
           "        option.dataset.displayModel = alias;\n"
           "        option.textContent = alias && canonical && alias !== canonical\n"
           "          ? `${alias} (${canonical})`\n"
           "          : (alias || canonical || '');\n"
           "        if ((canonical && canonical === requestedValue) || (!canonical && alias === requestedValue) || (!hasMatch && alias === requestedValue)) {\n"
           "          option.selected = true;\n"
           "          hasMatch = true;\n"
           "          matchedDisplay = alias || canonical;\n"
           "        }\n"
           "        select.appendChild(option);\n"
           "      });\n"
          "      if (availableModels.length === 0) {\n"
          "        placeholder.selected = true;\n"
          "        if (requestedValue) {\n"
          "          select.dataset.desiredModel = requestedValue;\n"
          "        } else {\n"
           "          delete select.dataset.desiredModel;\n"
           "        }\n"
           "        if (requestedDisplay) {\n"
           "          select.dataset.displayModel = requestedDisplay;\n"
           "        } else {\n"
           "          delete select.dataset.displayModel;\n"
           "        }\n"
          "      } else if (hasMatch) {\n"
          "        select.dataset.desiredModel = requestedValue;\n"
          "        select.dataset.displayModel = matchedDisplay || requestedDisplay || requestedValue;\n"
          "        if (requestedValue && suggestedValue === requestedValue) {\n"
          "          delete select.dataset.suggestedModel;\n"
          "        }\n"
          "      } else {\n"
          "        placeholder.selected = true;\n"
          "        if (requestedValue && !missingModelWarning && requestedValue !== suggestedValue) {\n"
          "          missingModelWarning = true;\n"
          "          if (!statusEl.textContent) {\n"
          "            setStatus('A previously selected model is no longer available.');\n"
          "          }\n"
           "        }\n"
           "        if (requestedValue) {\n"
           "          select.dataset.desiredModel = requestedValue;\n"
           "        } else {\n"
           "          delete select.dataset.desiredModel;\n"
           "        }\n"
           "        if (requestedDisplay) {\n"
           "          select.dataset.displayModel = requestedDisplay;\n"
           "        } else {\n"
           "          delete select.dataset.displayModel;\n"
           "        }\n"
           "      }\n"
           "    }\n"
          "    function registerModelSelect(select, selectedModel, isSuggestion) {\n"
          "      const suggestion = Boolean(isSuggestion && selectedModel);\n"
          "      if (selectedModel) {\n"
          "        select.dataset.desiredModel = selectedModel;\n"
          "        if (!select.dataset.displayModel) {\n"
          "          select.dataset.displayModel = selectedModel;\n"
          "        }\n"
          "        if (suggestion) {\n"
          "          select.dataset.suggestedModel = selectedModel;\n"
          "        } else {\n"
          "          delete select.dataset.suggestedModel;\n"
          "        }\n"
          "      } else {\n"
          "        delete select.dataset.desiredModel;\n"
          "        delete select.dataset.displayModel;\n"
          "        delete select.dataset.suggestedModel;\n"
          "      }\n"
          "      select.addEventListener('change', () => {\n"
          "        if (select.value) {\n"
          "          select.dataset.desiredModel = select.value;\n"
          "        } else {\n"
          "          delete select.dataset.desiredModel;\n"
          "        }\n"
           "        const currentOption = select.options[select.selectedIndex];\n"
           "        if (currentOption && currentOption.dataset && currentOption.dataset.displayModel) {\n"
           "          select.dataset.displayModel = currentOption.dataset.displayModel;\n"
           "        } else if (select.value) {\n"
           "          select.dataset.displayModel = select.value;\n"
          "        } else {\n"
          "          delete select.dataset.displayModel;\n"
          "        }\n"
          "        delete select.dataset.suggestedModel;\n"
          "        if (statusEl.textContent === 'A previously selected model is no longer available.') {\n"
          "          setStatus('');\n"
          "        }\n"
          "        updateStoredParticipants();\n"
          "      });\n"
          "      modelSelects.add(select);\n"
          "      populateModelOptions(select, selectedModel);\n"
          "      updateStoredParticipants();\n"
          "    }\n"
           "    function unregisterModelSelect(select) {\n"
            "      modelSelects.delete(select);\n"
            "    }\n"
            "    function refreshModelSelects() {\n"
            "      modelSelects.forEach((select) => {\n"
           "        const desired = select.dataset.desiredModel || select.value;\n"
           "        populateModelOptions(select, desired);\n"
            "      });\n"
            "    }\n"
           "    async function loadModels() {\n"
           "      missingModelWarning = false;\n"
           "      setConnectionState('checking');\n"
           "      if (statusEl.textContent === 'A previously selected model is no longer available.') {\n"
           "        setStatus('');\n"
           "      }\n"
           "      try {\n"
           "        const response = await fetch('/models');\n"
           "        if (!response.ok) {\n"
           "          throw new Error('Request failed');\n"
           "        }\n"
           "        const payload = await response.json();\n"
           "        availableModels = Array.isArray(payload.models) ? payload.models : [];\n"
           "        modelLoadError = false;\n"
           "        if (availableModels.length && statusEl.dataset.statusKey === MODEL_LOAD_ERROR_KEY) {\n"
           "          setStatus('');\n"
           "        }\n"
           "        if (availableModels.length > 0) {\n"
           "          setConnectionState('connected', 'API connected');\n"
           "        } else {\n"
           "          setConnectionState('connected', 'API connected (no models found)');\n"
           "        }\n"
           "      } catch (error) {\n"
           "        availableModels = [];\n"
           "        modelLoadError = true;\n"
           "        setConnectionState('disconnected', 'API unreachable');\n"
           "        if (!statusEl.textContent || statusEl.dataset.statusKey === MODEL_LOAD_ERROR_KEY) {\n"
           "          setStatus(MODEL_LOAD_ERROR_MESSAGE, MODEL_LOAD_ERROR_KEY);\n"
           "        }\n"
           "      }\n"
          "      refreshModelSelects();\n"
          "      updateStoredParticipants();\n"
          "    }\n"
          "    function createParticipant(name, model, isSuggestion, displayModel) {\n"
          "      const wrapper = document.createElement('div');\n"
          "      wrapper.className = 'participant';\n"
          "      wrapper.innerHTML = `\n"
          "        <label>Friendly name</label>\n"
          "        <input name=\"name\" placeholder=\"Astra\" value=\"${name || ''}\" />\n"
          "        <label>Open WebUI model</label>\n"
          "        <select name=\"model\"></select>\n"
          "        <button type=\"button\" class=\"remove\">Remove</button>\n"
          "      `;\n"
          "      const select = wrapper.querySelector('select[name=\"model\"]');\n"
          "      if (select) {\n"
          "        if (displayModel) {\n"
          "          select.dataset.displayModel = displayModel;\n"
          "        } else {\n"
          "          delete select.dataset.displayModel;\n"
          "        }\n"
          "      }\n"
          "      registerModelSelect(select, model || '', isSuggestion);\n"
          "      const nameInput = wrapper.querySelector('input[name=\"name\"]');\n"
          "      if (nameInput) {\n"
          "        nameInput.addEventListener('input', () => {\n"
          "          updateStoredParticipants();\n"
          "        });\n"
          "        nameInput.addEventListener('change', () => {\n"
          "          updateStoredParticipants();\n"
          "        });\n"
          "      }\n"
          "      wrapper.querySelector('.remove').addEventListener('click', () => {\n"
          "        unregisterModelSelect(select);\n"
          "        participantsEl.removeChild(wrapper);\n"
          "        updateParticipantCardThemes();\n"
          "        updateStoredParticipants();\n"
          "      });\n"
          "      participantsEl.appendChild(wrapper);\n"
          "      updateParticipantCardThemes();\n"
          "      updateStoredParticipants();\n"
          "    }\n"
          "    document.getElementById('addParticipant').addEventListener('click', (event) => {\n"
          "      event.preventDefault();\n"
          "      createParticipant('', '', false);\n"
          "    });\n"
           "    document.getElementById('start').addEventListener('click', async (event) => {\n"
           "      event.preventDefault();\n"
          "      setStatus('');\n"
           "      messagesEl.innerHTML = '';\n"
           "      participantStyles.clear();\n"
           "      transcriptEl.style.display = 'none';\n"
           "      const topic = document.getElementById('topic').value.trim();\n"
           "      const turns = parseInt(document.getElementById('turns').value, 10);\n"
           "      const participantDivs = participantsEl.querySelectorAll('.participant');\n"
           "      const participants = [];\n"
           "      participantDivs.forEach((div, index) => {\n"
           "        const name = div.querySelector('input[name=\"name\"]').value.trim();\n"
           "        const selectEl = div.querySelector('select[name=\"model\"]');\n"
           "        const rawValue = (selectEl && typeof selectEl.value === 'string') ? selectEl.value.trim() : '';\n"
           "        const selectedOption = selectEl ? selectEl.options[selectEl.selectedIndex] : null;\n"
           "        const optionDisplay = (selectedOption && selectedOption.dataset && typeof selectedOption.dataset.displayModel === 'string')\n"
           "          ? selectedOption.dataset.displayModel.trim()\n"
           "          : '';\n"
           "        const optionCanonical = (selectedOption && selectedOption.dataset && typeof selectedOption.dataset.canonicalModel === 'string')\n"
           "          ? selectedOption.dataset.canonicalModel.trim()\n"
           "          : '';\n"
           "        const datasetDisplay = (selectEl && selectEl.dataset && typeof selectEl.dataset.displayModel === 'string')\n"
           "          ? selectEl.dataset.displayModel.trim()\n"
           "          : '';\n"
           "        const canonicalValue = optionCanonical || rawValue;\n"
           "        const displayValue = optionDisplay || datasetDisplay || canonicalValue;\n"
           "        if (canonicalValue) {\n"
           "          participants.push({\n"
           "            name: name || `Companion ${index + 1}`,\n"
           "            model: canonicalValue,\n"
           "            displayModel: displayValue\n"
           "          });\n"
           "        }\n"
           "      });\n"
          "      if (!topic) {\n"
          "        setStatus('Please provide a topic.');\n"
          "        return;\n"
          "      }\n"
          "      if (Number.isNaN(turns) || turns < 1) {\n"
          "        setStatus('Please provide a valid number of turns.');\n"
          "        return;\n"
          "      }\n"
          "      if (participants.length === 0) {\n"
          "        setStatus('Add at least one participant with a model selected.');\n"
          "        return;\n"
          "      }\n"
           "      persistParticipantsState(participants);\n"
           "      resetTranscriptState(topic, turns, participants);\n"
           "      setStatus('Starting conversation...');\n"
           "      searchState.active = false;\n"
           "      if (enableSearchInput) {\n"
           "        searchState.requested = enableSearchInput.checked;\n"
           "      }\n"
           "      updateSearchHelper();\n"
           "      try {\n"
           "        const response = await fetch('/chat', {\n"
           "          method: 'POST',\n"
           "          headers: { 'Content-Type': 'application/json' },\n"
           "          body: JSON.stringify({ topic, turns, participants, enableSearch: enableSearchInput ? enableSearchInput.checked : false })\n"
           "        });\n"
          "        if (!response.ok) {\n"
          "          let payload = null;\n"
          "          try {\n"
          "            payload = await response.json();\n"
          "          } catch (parseError) {\n"
          "            // ignore JSON parse errors\n"
          "          }\n"
          "          const errorText = payload && typeof payload.error === 'string' && payload.error\n"
          "            ? payload.error\n"
          "            : 'The conversation failed.';\n"
          "          setStatus(errorText);\n"
          "          return;\n"
          "        }\n"
          "        const reader = response.body && response.body.getReader ? response.body.getReader() : null;\n"
          "        if (!reader) {\n"
          "          setStatus('Streaming is not supported by this browser.');\n"
          "          return;\n"
          "        }\n"
          "        setStatus('Waiting for responses...');\n"
           "        transcriptEl.style.display = 'block';\n"
           "        const decoder = new TextDecoder();\n"
           "        let buffer = '';\n"
           "        let stopStreaming = false;\n"
           "        let encounteredError = false;\n"
           "        while (!stopStreaming) {\n"
           "          const { value, done } = await reader.read();\n"
           "          if (done) {\n"
           "            break;\n"
           "          }\n"
           "          buffer += decoder.decode(value, { stream: true });\n"
           "          const lines = buffer.split('\\n');\n"
           "          buffer = lines.pop();\n"
           "          for (const line of lines) {\n"
           "            const trimmed = line.trim();\n"
           "            if (!trimmed) {\n"
           "              continue;\n"
           "            }\n"
           "            let eventPayload;\n"
           "            try {\n"
           "              eventPayload = JSON.parse(trimmed);\n"
           "            } catch (parseError) {\n"
           "              continue;\n"
           "            }\n"
          "            if (eventPayload.type === 'start') {\n"
          "              const participantsList = Array.isArray(eventPayload.participants) ? eventPayload.participants : [];\n"
          "              assignParticipantStyles(participantsList);\n"
          "              currentParticipants = normaliseParticipants(participantsList);\n"
          "              if (typeof eventPayload.topic === 'string') {\n"
          "                currentTopic = eventPayload.topic;\n"
          "              }\n"
          "              if (Number.isFinite(eventPayload.turns)) {\n"
          "                currentTurns = eventPayload.turns;\n"
          "              }\n"
           "              if (typeof eventPayload.searchRequested === 'boolean') {\n"
           "                searchState.requested = eventPayload.searchRequested;\n"
           "                conversationMetadata.searchRequested = eventPayload.searchRequested;\n"
           "                if (enableSearchInput) {\n"
           "                  enableSearchInput.checked = eventPayload.searchRequested;\n"
           "                }\n"
           "              }\n"
           "              if (typeof eventPayload.searchEnabled === 'boolean') {\n"
           "                const enabled = Boolean(eventPayload.searchEnabled);\n"
           "                searchState.active = enabled;\n"
           "                conversationMetadata.searchEnabled = enabled;\n"
           "              } else {\n"
           "                searchState.active = false;\n"
           "                conversationMetadata.searchEnabled = false;\n"
           "              }\n"
           "              updateSearchHelper();\n"
           "              setStatus('Conversation in progress...');\n"
          "            } else if (eventPayload.type === 'message' && eventPayload.message) {\n"
          "              appendMessage(eventPayload.message);\n"
          "              setStatus(`Responding: ${eventPayload.message.name}`);\n"
           "            } else if (eventPayload.type === 'error') {\n"
           "              const errorMessage = eventPayload.message && typeof eventPayload.message === 'string' && eventPayload.message\n"
           "                ? eventPayload.message\n"
           "                : 'The conversation failed.';\n"
           "              searchState.active = false;\n"
           "              if (typeof eventPayload.searchEnabled === 'boolean') {\n"
           "                conversationMetadata.searchEnabled = Boolean(eventPayload.searchEnabled);\n"
           "              }\n"
           "              if (enableSearchInput) {\n"
           "                searchState.requested = enableSearchInput.checked;\n"
           "              }\n"
           "              updateSearchHelper();\n"
           "              encounteredError = true;\n"
           "              setStatus(errorMessage);\n"
           "              stopStreaming = true;\n"
           "              break;\n"
           "            } else if (eventPayload.type === 'complete') {\n"
           "              if (typeof eventPayload.topic === 'string') {\n"
           "                currentTopic = eventPayload.topic;\n"
           "              }\n"
           "              if (Number.isFinite(eventPayload.turns)) {\n"
           "                currentTurns = eventPayload.turns;\n"
           "              }\n"
           "              searchState.active = false;\n"
           "              if (enableSearchInput) {\n"
           "                searchState.requested = enableSearchInput.checked;\n"
           "              }\n"
           "              updateSearchHelper();\n"
           "              summariseDiscussionIfNeeded();\n"
           "              setStatus('Conversation complete.');\n"
           "              stopStreaming = true;\n"
           "              break;\n"
          "            }\n"
           "          }\n"
           "          if (stopStreaming) {\n"
           "            await reader.cancel().catch(() => {});\n"
           "            break;\n"
           "          }\n"
           "        }\n"
           "        if (!stopStreaming) {\n"
           "          buffer += decoder.decode();\n"
           "          const trimmed = buffer.trim();\n"
          "          if (trimmed) {\n"
          "            try {\n"
          "              const eventPayload = JSON.parse(trimmed);\n"
          "              if (eventPayload.type === 'error') {\n"
          "                const errorMessage = eventPayload.message && typeof eventPayload.message === 'string' && eventPayload.message\n"
          "                  ? eventPayload.message\n"
          "                  : 'The conversation failed.';\n"
          "                encounteredError = true;\n"
          "                setStatus(errorMessage);\n"
          "              } else if (eventPayload.type === 'complete') {\n"
          "                if (typeof eventPayload.topic === 'string') {\n"
          "                  currentTopic = eventPayload.topic;\n"
          "                }\n"
          "                if (Number.isFinite(eventPayload.turns)) {\n"
          "                  currentTurns = eventPayload.turns;\n"
          "                }\n"
          "                if (typeof eventPayload.searchEnabled === 'boolean') {\n"
          "                  conversationMetadata.searchEnabled = Boolean(eventPayload.searchEnabled);\n"
          "                }\n"
          "                summariseDiscussionIfNeeded();\n"
          "                setStatus('Conversation complete.');\n"
          "              }\n"
          "            } catch (parseError) {\n"
           "              // ignore trailing parse issues\n"
           "            }\n"
           "          }\n"
           "        }\n"
           "        if (!encounteredError) {\n"
           "          summariseDiscussionIfNeeded();\n"
           "        }\n"
           "        searchState.active = false;\n"
           "        if (enableSearchInput) {\n"
           "          searchState.requested = enableSearchInput.checked;\n"
           "        }\n"
           "        updateSearchHelper();\n"
           "      } catch (error) {\n"
           "        setStatus('Unable to reach the aiChat server.');\n"
           "        searchState.active = false;\n"
           "        if (enableSearchInput) {\n"
           "          searchState.requested = enableSearchInput.checked;\n"
           "        }\n"
           "        updateSearchHelper();\n"
           "      }\n"
           "    });\n"
          "    const savedParticipants = loadStoredParticipants();\n"
          "    if (savedParticipants.length > 0) {\n"
          "      savedParticipants.forEach((participant) => {\n"
          "        const entry = participant || {};\n"
          "        const name = typeof entry.name === 'string' ? entry.name : '';\n"
          "        const model = typeof entry.model === 'string' ? entry.model : '';\n"
          "        const displayModel = typeof entry.displayModel === 'string' ? entry.displayModel : '';\n"
          "        createParticipant(name, model, false, displayModel);\n"
          "      });\n"
          "    } else {\n"
          "      createParticipant('Astra', 'gemma:2b', true);\n"
          "      createParticipant('Nova', 'llama3:8b', true);\n"
          "    }\n"
          "    loadModels();\n"
           "  </script>\n"
           "</body>\n"
           "</html>\n";
}

static char *render_html_page(void) {
    const char *template = get_html_page();
    size_t template_len = strlen(template);
    char *copy = malloc(template_len + 1);

    if (!copy) {
        return NULL;
    }

    memcpy(copy, template, template_len + 1);
    return copy;
}

static int parse_int_header(const char *headers, const char *key) {
    const char *location = strcasestr(headers, key);
    if (!location) {
        return -1;
    }
    location += strlen(key);
    while (*location && isspace((unsigned char)*location)) {
        location++;
    }
    return atoi(location);
}

static int read_http_request(int client_fd, char **out_request, size_t *out_length) {
    size_t capacity = READ_BUFFER_CHUNK;
    size_t length = 0;
    char *buffer = malloc(capacity);
    if (!buffer) {
        return -1;
    }

    while (1) {
        ssize_t bytes = recv(client_fd, buffer + length, capacity - length, 0);
        if (bytes <= 0) {
            free(buffer);
            return -1;
        }
        length += (size_t)bytes;

        char *header_end = memmem(buffer, length, "\r\n\r\n", 4);
        if (header_end) {
            size_t header_length = (size_t)(header_end - buffer) + 4;
            int content_length = parse_int_header(buffer, "Content-Length:");
            size_t total_length = header_length + (content_length > 0 ? (size_t)content_length : 0);
            while (length < total_length) {
                if (length == capacity) {
                    capacity *= 2;
                    char *tmp = realloc(buffer, capacity);
                    if (!tmp) {
                        free(buffer);
                        return -1;
                    }
                    buffer = tmp;
                }
                bytes = recv(client_fd, buffer + length, capacity - length, 0);
                if (bytes <= 0) {
                    free(buffer);
                    return -1;
                }
                length += (size_t)bytes;
            }
            *out_request = buffer;
            *out_length = length;
            return 0;
        }

        if (length == capacity) {
            capacity *= 2;
            char *tmp = realloc(buffer, capacity);
            if (!tmp) {
                free(buffer);
                return -1;
            }
            buffer = tmp;
        }
    }
}

static void handle_chat_request(int client_fd, const char *body, size_t body_length, const char *ollama_url) {
    json_object *payload = NULL;
    json_object *topic_obj = NULL;
    json_object *turns_obj = NULL;
    json_object *participants_obj = NULL;
    const char *topic = NULL;
    int turns = 0;
    struct Participant participants[MAX_PARTICIPANTS];
    size_t participant_count = 0;
    json_object *result = NULL;
    char *error_message = NULL;
    int search_requested = 0;
    int search_enabled = 0;
    json_object *search_flag_obj = NULL;

    memset(participants, 0, sizeof(participants));
    struct json_tokener *tok = json_tokener_new();
    if (!tok) {
        send_http_error(client_fd, "500 Internal Server Error", "Unable to initialise JSON parser.");
        return;
    }

    payload = json_tokener_parse_ex(tok, body, (int)body_length);
    if (json_tokener_get_error(tok) != json_tokener_success || !payload) {
        json_tokener_free(tok);
        send_http_error(client_fd, "400 Bad Request", "Invalid JSON payload.");
        return;
    }
    json_tokener_free(tok);

    if (!json_object_object_get_ex(payload, "topic", &topic_obj) ||
        json_object_get_type(topic_obj) != json_type_string) {
        json_object_put(payload);
        send_http_error(client_fd, "400 Bad Request", "Field 'topic' is required.");
        return;
    }
    topic = json_object_get_string(topic_obj);

    if (!json_object_object_get_ex(payload, "turns", &turns_obj)) {
        json_object_put(payload);
        send_http_error(client_fd, "400 Bad Request", "Field 'turns' is required.");
        return;
    }
    turns = json_object_get_int(turns_obj);
    if (turns < MIN_TURNS) {
        turns = MIN_TURNS;
    }
    if (turns > MAX_TURNS) {
        turns = MAX_TURNS;
    }

    if (!json_object_object_get_ex(payload, "participants", &participants_obj) ||
        json_object_get_type(participants_obj) != json_type_array) {
        json_object_put(payload);
        send_http_error(client_fd, "400 Bad Request", "Field 'participants' must be an array.");
        return;
    }

    size_t array_len = json_object_array_length(participants_obj);
    if (array_len == 0) {
        json_object_put(payload);
        send_http_error(client_fd, "400 Bad Request", "Provide at least one participant.");
        return;
    }
    if (array_len > MAX_PARTICIPANTS) {
        array_len = MAX_PARTICIPANTS;
    }

    for (size_t i = 0; i < array_len; ++i) {
        json_object *item = json_object_array_get_idx(participants_obj, i);
        json_object *name_obj = NULL;
        json_object *model_obj = NULL;
        json_object *display_obj = NULL;
        const char *name = NULL;
        const char *model = NULL;
        const char *display = NULL;

        if (!item || json_object_get_type(item) != json_type_object) {
            continue;
        }

        if (json_object_object_get_ex(item, "model", &model_obj) &&
            json_object_get_type(model_obj) == json_type_string) {
            model = json_object_get_string(model_obj);
        }
        if (!model || !*model) {
            continue;
        }

        if (json_object_object_get_ex(item, "displayModel", &display_obj) &&
            json_object_get_type(display_obj) == json_type_string && json_object_get_string_len(display_obj) > 0) {
            display = json_object_get_string(display_obj);
        }

        if (json_object_object_get_ex(item, "name", &name_obj) &&
            json_object_get_type(name_obj) == json_type_string && json_object_get_string_len(name_obj) > 0) {
            name = json_object_get_string(name_obj);
        }
        if (!name || !*name) {
            static const char *fallback_names[] = {"Astra", "Nova", "Cosmo", "Lyric", "Echo", "Muse"};
            size_t fallback_idx = participant_count < (sizeof(fallback_names) / sizeof(fallback_names[0]))
                                      ? participant_count
                                      : participant_count % (sizeof(fallback_names) / sizeof(fallback_names[0]));
            name = fallback_names[fallback_idx];
        }

        strncpy(participants[participant_count].name, name, MAX_NAME_LENGTH - 1);
        participants[participant_count].name[MAX_NAME_LENGTH - 1] = '\0';
        strncpy(participants[participant_count].model, model, MAX_MODEL_LENGTH - 1);
        participants[participant_count].model[MAX_MODEL_LENGTH - 1] = '\0';
        if (display && *display) {
            strncpy(participants[participant_count].display_model, display, MAX_MODEL_LENGTH - 1);
            participants[participant_count].display_model[MAX_MODEL_LENGTH - 1] = '\0';
        }
        participant_count++;
    }

    if (json_object_object_get_ex(payload, "enableSearch", &search_flag_obj) && search_flag_obj) {
        search_requested = json_object_get_boolean(search_flag_obj);
    }

    json_object_put(payload);
    search_enabled = search_requested;

    int needs_lookup = 0;
    for (size_t i = 0; i < participant_count; ++i) {
        if (participants[i].display_model[0] == '\0') {
            needs_lookup = 1;
            break;
        }
    }

    json_object *models_payload = NULL;
    if (needs_lookup && fetch_available_models(ollama_url, &models_payload, NULL) != 0) {
        models_payload = NULL;
    }
    ensure_participant_display_models(participants, participant_count, models_payload);
    if (models_payload) {
        json_object_put(models_payload);
    }

    if (participant_count == 0) {
        send_http_error(client_fd, "400 Bad Request", "No valid participants supplied.");
        return;
    }

    if (send_chunked_header(client_fd, "200 OK", "application/x-ndjson") != 0) {
        return;
    }

    json_object *start_event = json_object_new_object();
    json_object *start_participants = json_object_new_array();
    if (!start_event || !start_participants) {
        if (start_event) {
            json_object_put(start_event);
        }
        if (start_participants) {
            json_object_put(start_participants);
        }
        send_stream_error_event(client_fd, "Failed to start stream.");
        finish_chunked_response(client_fd);
        return;
    }

    for (size_t i = 0; i < participant_count; ++i) {
        json_object *participant_obj = json_object_new_object();
        if (!participant_obj) {
            json_object_put(start_participants);
            json_object_put(start_event);
            send_stream_error_event(client_fd, "Failed to start stream.");
            finish_chunked_response(client_fd);
            return;
        }
        json_object_object_add(participant_obj, "name", json_object_new_string(participants[i].name));
        json_object_object_add(participant_obj, "model", json_object_new_string(participants[i].model));
        if (participants[i].display_model[0] != '\0') {
            json_object_object_add(participant_obj, "displayModel",
                                   json_object_new_string(participants[i].display_model));
        }
        json_object_array_add(start_participants, participant_obj);
    }

    json_object_object_add(start_event, "type", json_object_new_string("start"));
    json_object_object_add(start_event, "topic", json_object_new_string(topic));
    json_object_object_add(start_event, "turns", json_object_new_int(turns));
    json_object_object_add(start_event, "participants", start_participants);
    json_object_object_add(start_event, "searchRequested", json_object_new_boolean(search_requested));
    json_object_object_add(start_event, "searchEnabled", json_object_new_boolean(search_enabled));

    if (send_json_chunk(client_fd, start_event) != 0) {
        json_object_put(start_event);
        finish_chunked_response(client_fd);
        return;
    }
    json_object_put(start_event);

    struct StreamContext stream_ctx = {client_fd, 0};
    if (run_conversation(topic, turns, participants, participant_count, ollama_url, search_enabled,
                         stream_message_callback,
                         &stream_ctx, &result, &error_message) != 0) {
        if (!stream_ctx.failed) {
            if (error_message) {
                send_stream_error_event(client_fd, error_message);
            } else {
                send_stream_error_event(client_fd, "Conversation failed.");
            }
            finish_chunked_response(client_fd);
        }
        if (error_message) {
            free(error_message);
        }
        if (result) {
            json_object_put(result);
        }
        return;
    }

    if (!stream_ctx.failed) {
        json_object *complete_event = json_object_new_object();
        if (complete_event) {
            json_object_object_add(complete_event, "type", json_object_new_string("complete"));
            json_object_object_add(complete_event, "topic", json_object_new_string(topic));
            json_object_object_add(complete_event, "turns", json_object_new_int(turns));
            json_object_object_add(complete_event, "searchEnabled",
                                   json_object_new_boolean(search_enabled));
            if (send_json_chunk(client_fd, complete_event) != 0) {
                stream_ctx.failed = 1;
            }
            json_object_put(complete_event);
        }
    }

    if (!stream_ctx.failed) {
        finish_chunked_response(client_fd);
    }

    if (result) {
        json_object_put(result);
    }
    if (error_message) {
        free(error_message);
    }
}

static void handle_client(int client_fd, const char *ollama_url) {
    char *request = NULL;
    size_t request_len = 0;
    char method[8] = {0};
    char path[64] = {0};
    char *body = NULL;
    size_t body_length = 0;

    if (read_http_request(client_fd, &request, &request_len) != 0) {
        send_http_error(client_fd, "400 Bad Request", "Unable to read request.");
        return;
    }

    sscanf(request, "%7s %63s", method, path);

    char *separator = strstr(request, "\r\n\r\n");
    if (separator) {
        body = separator + 4;
        body_length = request_len - (size_t)(body - request);
    }

    if (strcmp(method, "GET") == 0 && strcmp(path, "/") == 0) {
        char *html = render_html_page();
        if (!html) {
            send_http_error(client_fd, "500 Internal Server Error", "Unable to render interface.");
        } else {
            send_http_response(client_fd, "200 OK", "text/html; charset=UTF-8", html);
            free(html);
        }
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/models") == 0) {
        handle_models_request(client_fd, ollama_url);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/diagnostics") == 0) {
        handle_diagnostics_request(client_fd, ollama_url);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/chat") == 0) {
        if (!body) {
            send_http_error(client_fd, "400 Bad Request", "Missing request body.");
        } else {
            handle_chat_request(client_fd, body, body_length, ollama_url);
        }
    } else if (strcmp(method, "OPTIONS") == 0) {
        const char *response =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "Connection: close\r\n\r\n";
        send(client_fd, response, strlen(response), 0);
    } else {
        send_http_error(client_fd, "404 Not Found", "Endpoint not found.");
    }

    free(request);
}

static int probe_models_endpoint(const char *label, const char *url) {
    json_object *payload = NULL;
    json_object *models = NULL;
    char *error_message = NULL;
    int rc = -1;

    if (!url) {
        printf("[SKIP] %s: endpoint not available.\n", label);
        return -1;
    }

    printf("[TEST] %s\n", label);
    printf("        %s\n", url);

    rc = fetch_models_via_url(url, &payload, &error_message);
    if (rc == 0) {
        int model_count = 0;
        if (json_object_object_get_ex(payload, "models", &models) &&
            json_object_is_type(models, json_type_array)) {
            model_count = (int)json_object_array_length(models);
        }
        printf("[PASS] Received %d model%s from %s.\n",
               model_count,
               model_count == 1 ? "" : "s",
               label);
        json_object_put(payload);
        return 0;
    }

    printf("[FAIL] %s request failed: %s\n",
           label,
           error_message ? error_message : "see stderr for details");
    if (error_message) {
        free(error_message);
    }
    if (payload) {
        json_object_put(payload);
    }

    return -1;
}

static int print_webui_diagnostics(void) {
    const char *ollama_url = get_ollama_url();
    const char *api_key = get_webui_key();
    char *models_url = build_webui_models_url(ollama_url);
    char *fallback_url = build_ollama_tags_url(ollama_url);
    int success = 0;

    printf("Open WebUI diagnostics\n");
    printf("------------------------\n");
    printf("Base URL: %s\n", ollama_url);
    printf("API key: %s\n", api_key ? "[SET]" : "(not provided)");

    if (probe_models_endpoint("Open WebUI models endpoint", models_url) == 0) {
        success = 1;
    }

    if (probe_models_endpoint("Legacy Ollama fallback endpoint", fallback_url) == 0) {
        success = 1;
    }

    if (!success) {
        printf("No endpoints responded successfully. Review the error details above to continue troubleshooting.\n");
    }

    free(models_url);
    free(fallback_url);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

static void print_usage(const char *program_name) {
    printf("Usage: %s [--check-webui]\n", program_name ? program_name : "openaichat");
    printf("  --check-webui    Probe the Open WebUI API endpoints and exit.\n");
}

int main(int argc, char *argv[]) {
    int server_fd = -1;
    struct sockaddr_in address;
    int opt = 1;
    int port = DEFAULT_PORT;
    int requested_port = DEFAULT_PORT;
    int fallback_used = 0;
    int port_from_env = 0;
    const char *port_env = getenv("AICHAT_PORT");
    const char *ollama_url = get_ollama_url();
    /* *** ADDED FOR OPEN WEBUI *** */
    const char *api_key = get_webui_key();

    if (argc > 1) {
        if (strcmp(argv[1], "--check-webui") == 0) {
            return print_webui_diagnostics();
        }
        if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        fprintf(stderr, "Unknown argument: %s\n", argv[1]);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (port_env && *port_env) {
        char *endptr = NULL;
        long parsed = strtol(port_env, &endptr, 10);
        if (endptr && *endptr == '\0' && parsed > 0 && parsed < 65535) {
            port = (int)parsed;
            port_from_env = 1;
        } else {
            fprintf(stderr, "Warning: invalid AICHAT_PORT '%s', using default %d.\n", port_env, DEFAULT_PORT);
        }
    }
    requested_port = port;

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        perror("socket");
        return EXIT_FAILURE;
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt");
        close(server_fd);
        return EXIT_FAILURE;
    }

    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;

    for (int attempt = 0; attempt <= FALLBACK_PORT_STEPS; ++attempt) {
        address.sin_port = htons((uint16_t)port);
        if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) == 0) {
            if (attempt > 0) {
                fallback_used = 1;
            }
            break;
        }

        if (!(errno == EADDRINUSE && !port_from_env && attempt < FALLBACK_PORT_STEPS)) {
            perror("bind");
            close(server_fd);
            return EXIT_FAILURE;
        }

        int next_port = DEFAULT_PORT + attempt + 1;
        fprintf(stderr, "Port %d unavailable, trying %d instead.\n", port, next_port);
        port = next_port;
    }

    socklen_t addrlen = sizeof(address);
    if (getsockname(server_fd, (struct sockaddr *)&address, &addrlen) < 0) {
        perror("getsockname");
        close(server_fd);
        return EXIT_FAILURE;
    }
    port = ntohs(address.sin_port);

    if (listen(server_fd, 10) < 0) {
        perror("listen");
        close(server_fd);
        return EXIT_FAILURE;
    }

    if (fallback_used) {
        printf("Port %d unavailable, using fallback port %d.\n", requested_port, port);
    }

    printf("aiChat web server ready on http://127.0.0.1:%d\n", port);
    printf("Using Open WebUI endpoint: %s\n", ollama_url);

    char *diagnostics_models_url = build_webui_models_url(ollama_url);
    if (diagnostics_models_url) {
        printf("Model list will be requested from: %s\n", diagnostics_models_url);
        free(diagnostics_models_url);
    } else {
        fprintf(stderr, "Warning: unable to derive Open WebUI models endpoint from %s.\n", ollama_url);
    }

    char *diagnostics_fallback_url = build_ollama_tags_url(ollama_url);
    if (diagnostics_fallback_url) {
        printf("Legacy fallback endpoint: %s\n", diagnostics_fallback_url);
        free(diagnostics_fallback_url);
    }

    /* *** ADDED FOR OPEN WEBUI *** */
    if (!api_key) {
        fprintf(stderr,
                "WEBUI_API_KEY not set in config.h or the environment; continuing without authentication. Configure it if your server requires a token.\n");
    } else {
        printf("Using Open WebUI API Key: [SET]\n");
    }

    while (1) {
        int client_fd;
        socklen_t addrlen = sizeof(address);
        client_fd = accept(server_fd, (struct sockaddr *)&address, &addrlen);
        if (client_fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("accept");
            break;
        }

        handle_client(client_fd, ollama_url);
        shutdown(client_fd, SHUT_RDWR);
        close(client_fd);
    }

    close(server_fd);
    return EXIT_SUCCESS;
}
