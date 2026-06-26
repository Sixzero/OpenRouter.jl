using Test
using OpenRouter
using OpenRouter: get_provider_info, extract_provider_from_model, OllamaSchema,
    build_url, build_payload, build_messages, extract_content, extract_reasoning,
    extract_finish_reason, extract_tool_calls, extract_tokens, extract_chunks,
    is_done, is_start, is_sse_stream, Tool, list_native_models

@testset "Ollama Provider Basic" begin
    info = get_provider_info("ollama")
    @test info !== nothing
    @test info.base_url == "http://localhost:11434/api"
    @test info.schema isa OllamaSchema
    @test info.api_key_env_var === nothing

    cloud = get_provider_info("ollama_cloud")
    @test cloud !== nothing
    @test cloud.base_url == "https://ollama.com/api"
    @test cloud.schema isa OllamaSchema
    @test cloud.api_key_env_var == "OLLAMA_API_KEY"

    # Model parsing with double colon
    @test extract_provider_from_model("ollama:smollm:360m") == "ollama"
    @test extract_provider_from_model("ollama_cloud:gpt-oss:20b") == "ollama_cloud"
end

@testset "OllamaSchema payload/url" begin
    s = OllamaSchema()
    @test build_url(s, "http://localhost:11434/api", "x", false) == "http://localhost:11434/api/chat"
    @test build_url(s, "http://localhost:11434/api", "x", true) == "http://localhost:11434/api/chat"

    p = build_payload(s, "hi", "llama3.2", "be brief", false; temperature=0.5, think=true)
    @test p["model"] == "llama3.2"
    @test p["stream"] == false
    @test p["think"] == true
    @test p["options"]["temperature"] == 0.5          # sampling params nested under options
    @test p["messages"][1]["role"] == "system"
    @test p["messages"][1]["content"] == "be brief"
    @test p["messages"][2]["role"] == "user"
    @test p["messages"][2]["content"] == "hi"

    # tools converted to OpenAI-style function schema
    tool = Tool(name="f", description="d", parameters=Dict("type"=>"object"))
    pt = build_payload(s, "hi", "m", nothing, false; tools=[tool])
    @test pt["tools"][1]["type"] == "function"
    @test pt["tools"][1]["function"]["name"] == "f"
end

@testset "OllamaSchema image messages" begin
    s = OllamaSchema()
    # 1x1 transparent PNG
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    msg = OpenRouter.UserMessage(content="what is this?", image_data=["data:image/png;base64,$png"])
    built = build_messages(s, [msg], nothing)
    @test built[1]["content"] == "what is this?"
    @test built[1]["images"] == [png]   # bare base64, no data: prefix
end

@testset "OllamaSchema response extraction" begin
    s = OllamaSchema()
    r = Dict(
        "message" => Dict("role"=>"assistant", "content"=>"Hello", "thinking"=>"hmm"),
        "done" => true, "done_reason" => "stop",
        "prompt_eval_count" => 5, "eval_count" => 3,
    )
    @test extract_content(s, r) == "Hello"
    @test extract_reasoning(s, r) == "hmm"
    @test extract_finish_reason(s, r) == "stop"
    tok = extract_tokens(s, r)
    @test tok.prompt_tokens == 5
    @test tok.completion_tokens == 3
    @test tok.total_tokens == 8
    @test extract_tokens(s, Dict("message"=>Dict("content"=>"x"))) === nothing

    rt = Dict("message" => Dict("role"=>"assistant", "content"=>"",
        "tool_calls" => [Dict("function"=>Dict("name"=>"f", "arguments"=>Dict("a"=>1)))]))
    tcs = extract_tool_calls(s, rt)
    @test tcs !== nothing
    @test tcs[1]["function"]["name"] == "f"
    @test tcs[1]["type"] == "function"
end

@testset "OllamaSchema NDJSON streaming" begin
    s = OllamaSchema()
    @test is_sse_stream(s) == false

    blob = "{\"message\":{\"role\":\"assistant\",\"content\":\"a\"},\"done\":false}\n" *
           "{\"message\":{\"content\":\"b\"},\"done\":true,\"done_reason\":\"stop\"}\n"
    chunks, spill = extract_chunks(s, blob)
    @test length(chunks) == 2
    @test spill == ""
    @test is_start(s, chunks[1])
    @test !is_done(s, chunks[1])
    @test is_done(s, chunks[2])
    @test extract_content(s, chunks[1]) == "a"

    # partial line is held as spillover until completed
    c1, sp1 = extract_chunks(s, "{\"message\":{\"content\":\"x\"}")
    @test isempty(c1)
    @test sp1 == "{\"message\":{\"content\":\"x\"}"
    c2, sp2 = extract_chunks(s, ",\"done\":true}\n"; spillover=sp1)
    @test length(c2) == 1
    @test sp2 == ""
    @test extract_content(s, c2[1]) == "x"
end

# Live tests against Ollama Cloud — only run when OLLAMA_API_KEY is set.
if !isempty(get(ENV, "OLLAMA_API_KEY", ""))
    @testset "Ollama Cloud live" begin
        r = aigen("Say hello in exactly 2 words", "ollama_cloud:gpt-oss:20b"; think=false)
        @test !isempty(r.content)
        @test r.tokens.completion_tokens > 0

        cb = HttpStreamCallback(; out=devnull)
        rs = aigen("Count 1 to 3", "ollama_cloud:gpt-oss:20b"; think=false, streamcallback=cb)
        @test !isempty(rs.content)
        @test length(cb.chunks) > 1

        models = list_native_models("ollama_cloud")
        @test !isempty(models)
    end
else
    @info "Skipping Ollama Cloud live tests (set OLLAMA_API_KEY to enable)"
end
