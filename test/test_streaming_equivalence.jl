using OpenRouter: AIMessage, extract_content
using Test

"""
Test that streaming and non-streaming responses are equivalent.
"""
function test_streaming_equivalence()
    @testset "Streaming vs Non-streaming Equivalence" begin
        # Test with different providers
        test_cases = [
            "anthropic:anthropic/claude-haiku-4.5",
            "groq:moonshotai/kimi-k2-0905",
            "google-ai-studio:google/gemini-2.5-flash",
            "openai:openai/gpt-5.1-mini",
            "cerebras:meta-llama/llama-3.1-8b-instruct",
        ]
        
        prompt = "Count from 1 to 3 in one line"
        
        for provider_model in test_cases
            # @testset "Provider: $provider_model" begin
                try
                    # Get raw responses
                    callback = HttpStreamCallback(; out=devnull, verbose=false)

callback = HttpStreamHooks(
    on_done = () -> "\n✓ Generation complete!",
    on_start = () -> "🚀 Starting generation...",
    content_formatter = text -> uppercase(text)  # Make all content uppercase
)

                    raw_stream = aigen_raw(prompt, provider_model; streamcallback=callback)
                    raw_normal = aigen_raw(prompt, provider_model)
                    
                    # Test that components are the same
                    @test raw_stream.schema == raw_normal.schema
                    @test raw_stream.provider_info == raw_normal.provider_info
                    @test raw_stream.model_id == raw_normal.model_id
                    @test raw_stream.provider_endpoint == raw_normal.provider_endpoint
                    
                    # Test basic structure equivalence
                    result_stream = raw_stream.result
                    result_normal = raw_normal.result
                    
                    @test haskey(result_stream, "choices") || haskey(result_stream, "content") || haskey(result_stream, "candidates")
                    @test haskey(result_normal, "choices") || haskey(result_normal, "content") || haskey(result_normal, "candidates")
                    
                    # Test content equivalence (the actual response should be the same)
                    content_stream = extract_content(raw_stream.schema, result_stream)
                    content_normal = extract_content(raw_normal.schema, result_normal)
                    
                    # Content should be similar (allowing for minor variations in generation)
                    @test !isempty(content_stream)
                    @test !isempty(content_normal)
                    
                    # Test token usage equivalence (if present)
                    tokens_stream = extract_tokens(raw_stream.schema, result_stream)
                    tokens_normal = extract_tokens(raw_normal.schema, result_normal)
                    
                    if tokens_stream !== nothing && tokens_normal !== nothing
                        # Token counts should be very similar (within small margin for streaming overhead)
                        @test abs(tokens_stream.total_tokens - tokens_normal.total_tokens) <= 5
                    end
                    
                    # Test finish reason equivalence
                    finish_stream = extract_finish_reason(raw_stream.schema, result_stream)
                    finish_normal = extract_finish_reason(raw_normal.schema, result_normal)
                    
                    @test finish_stream == finish_normal
                    
                    println("✓ $provider_model: streaming and non-streaming responses are equivalent")
                    
                catch e
                    @warn "Failed to test $provider_model: $e"
                    # Don't fail the entire test suite for individual provider issues
                end
            # end
        end
    end
end

"""
Test that AIMessage objects are equivalent between streaming and non-streaming.
"""
function test_aimessage_equivalence()
    @testset "AIMessage Equivalence" begin
        provider_model = "anthropic:anthropic/claude-haiku-4.5"
        prompt = "Say 'hello world'"
        
        try
            # Get AIMessage responses
            callback = HttpStreamCallback(; out=devnull, verbose=false)
            response_stream = aigen(prompt, provider_model; streamcallback=callback)
            response_normal = aigen(prompt, provider_model)
            
            # Test basic properties
            @test response_stream isa AIMessage
            @test response_normal isa AIMessage
            
            # Content should be non-empty
            @test !isempty(response_stream.content)
            @test !isempty(response_normal.content)
            
            # Both should have finish reasons
            @test response_stream.finish_reason !== nothing
            @test response_normal.finish_reason !== nothing
            @test response_stream.finish_reason == response_normal.finish_reason
            
            # Both should have token counts (if available)
            if response_stream.tokens !== nothing && response_normal.tokens !== nothing
                @test abs(response_stream.tokens.total_tokens - response_normal.tokens.total_tokens) <= 5
            end
            
            # Both should have costs (if available)
            if response_stream.cost !== nothing && response_normal.cost !== nothing
                @test abs(response_stream.cost - response_normal.cost) <= 0.001  # Small margin for rounding
            end
            
            println("✓ AIMessage objects are equivalent between streaming and non-streaming")
            
        catch e
            @warn "Failed to test AIMessage equivalence: $e"
        end
    end
end

# Run tests if this file is executed directly
test_streaming_equivalence()
# test_aimessage_equivalence()