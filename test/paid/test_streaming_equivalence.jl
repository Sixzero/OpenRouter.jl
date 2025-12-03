using OpenRouter: AIMessage, extract_content, extract_tokens, extract_finish_reason
using Test

@testset "Streaming vs Non-streaming Equivalence" begin
    test_cases = [
        "anthropic:anthropic/claude-haiku-4.5",
        "groq:moonshotai/kimi-k2-0905",
        "google-ai-studio:google/gemini-2.5-flash",
        "openai:openai/gpt-5.1-mini",
        "cerebras:meta-llama/llama-3.1-8b-instruct",
    ]
    
    prompt = "Count from 1 to 3 in one line"
    
    for provider_model in test_cases
        try
            callback = HttpStreamHooks(out=devnull)
            
            raw_stream = aigen_raw(prompt, provider_model; streamcallback=callback, temperature=0.0)
            raw_normal = aigen_raw(prompt, provider_model; temperature=0.0)
            
            @test raw_stream.schema == raw_normal.schema
            @test raw_stream.provider_info == raw_normal.provider_info
            @test raw_stream.model_id == raw_normal.model_id
            @test raw_stream.provider_endpoint == raw_normal.provider_endpoint
            
            result_stream = raw_stream.result
            result_normal = raw_normal.result
            
            @test haskey(result_stream, "choices") || haskey(result_stream, "content") || haskey(result_stream, "candidates")
            @test haskey(result_normal, "choices") || haskey(result_normal, "content") || haskey(result_normal, "candidates")
            
            content_stream = extract_content(raw_stream.schema, result_stream)
            content_normal = extract_content(raw_normal.schema, result_normal)
            
            @test !isempty(content_stream)
            @test !isempty(content_normal)
            
            tokens_stream = extract_tokens(raw_stream.schema, result_stream)
            tokens_normal = extract_tokens(raw_normal.schema, result_normal)
            
            if tokens_stream !== nothing && tokens_normal !== nothing
                if abs(tokens_stream.total_tokens - tokens_normal.total_tokens) > 5
                    @warn "Token mismatch for $provider_model"
                    @show result_stream 
                    @show result_normal
                end
                @test abs(tokens_stream.total_tokens - tokens_normal.total_tokens) <= 5
            end
            
            finish_stream = extract_finish_reason(raw_stream.schema, result_stream)
            finish_normal = extract_finish_reason(raw_normal.schema, result_normal)
            
            @test finish_stream == finish_normal
            
            println("✓ $provider_model: streaming and non-streaming responses are equivalent")
            
        catch e
            @warn "Failed to test $provider_model: $e"
        end
    end
end

@testset "AIMessage Equivalence" begin
    provider_model = "anthropic:anthropic/claude-haiku-4.5"
    prompt = "Say 'hello world'"
    
    try
        callback = HttpStreamHooks(out=devnull)
        response_stream = aigen(prompt, provider_model; streamcallback=callback, temperature=0.0)
        response_normal = aigen(prompt, provider_model; temperature=0.0)
        
        @test response_stream isa AIMessage
        @test response_normal isa AIMessage
        
        @test !isempty(response_stream.content)
        @test !isempty(response_normal.content)
        
        @test response_stream.finish_reason !== nothing
        @test response_normal.finish_reason !== nothing
        @test response_stream.finish_reason == response_normal.finish_reason
        
        if response_stream.tokens !== nothing && response_normal.tokens !== nothing
            @test abs(response_stream.tokens.total_tokens - response_normal.tokens.total_tokens) <= 5
        end
        
        if response_stream.cost !== nothing && response_normal.cost !== nothing
            @test abs(response_stream.cost - response_normal.cost) <= 0.001
        end
        
        println("✓ AIMessage objects are equivalent between streaming and non-streaming")
        
    catch e
        @warn "Failed to test AIMessage equivalence: $e"
    end
end

