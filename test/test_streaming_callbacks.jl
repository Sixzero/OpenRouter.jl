using Test
using OpenRouter
using OpenRouter: HttpStreamHooks

# Shared callback state tracker
mutable struct CallbackTracker
    start_count::Int
    chunk_count::Int
    done_count::Int
    meta_ai_count::Int
    meta_usr_count::Int
    content::String
    
    CallbackTracker() = new(0, 0, 0, 0, 0, "")
end

function create_tracked_callback(tracker::CallbackTracker)
    return HttpStreamHooks(
        on_start = () -> begin
            tracker.start_count += 1
            "▶ Stream started"
        end,
        content_formatter = (content) -> begin
            tracker.chunk_count += 1
            tracker.content *= content
            content  # Return content unchanged
        end,
        on_done = () -> begin
            tracker.done_count += 1
            "\n✓ Stream completed"
        end,
        on_meta_ai = (tokens, cost, elapsed) -> begin
            tracker.meta_ai_count += 1
            "ℹ Meta AI received: tokens=$(tokens.total_tokens)"
        end,
        on_meta_usr = (tokens, cost, elapsed) -> begin
            tracker.meta_usr_count += 1
            "ℹ Meta USR received: tokens=$(tokens.prompt_tokens)"
        end
    )
end
@testset "Streaming Callbacks - Provider Coverage" begin
    
    prompt = "Count from 1 to 5 in one line:"
    
    @testset "ChatCompletion Schema (Groq/Moonshot)" begin
        tracker = CallbackTracker()
        callback = create_tracked_callback(tracker)
        
        response = aigen(prompt, "groq:moonshotai/kimi-k2-0905"; streamcallback=callback)
        
        @test tracker.start_count == 1
        @test tracker.chunk_count >= 1
        @test tracker.done_count == 1
        @test tracker.meta_ai_count == 1  # Should be called at least once with token info
        @test tracker.meta_usr_count == 0  # ChatCompletion doesn't separate user tokens
        @test !isempty(tracker.content)
        @test response.content == tracker.content
    end
    
    @testset "Anthropic Schema" begin
        tracker = CallbackTracker()
        callback = create_tracked_callback(tracker)
        
        response = aigen(prompt, "anthropic:anthropic/claude-haiku-4.5"; streamcallback=callback)
        
        @test tracker.start_count == 1
        @test tracker.chunk_count >= 1
        @test tracker.done_count == 1
        @test tracker.meta_ai_count >= 1  # AI tokens in message_delta
        @test tracker.meta_usr_count >= 1  # User tokens in message_start
        @test !isempty(tracker.content)
        @test response.content == tracker.content
    end
    
    @testset "Gemini Schema" begin
        tracker = CallbackTracker()
        callback = create_tracked_callback(tracker)
        
        response = aigen(prompt, "google-ai-studio:google/gemini-2.5-flash"; streamcallback=callback)
        
        @test tracker.start_count == 1
        @test tracker.chunk_count >= 1
        @test tracker.done_count == 1
        @test tracker.meta_ai_count == 1  # Should be called with token info
        @test tracker.meta_usr_count == 1  # Gemini doesn't separate user tokens
        @test !isempty(tracker.content)
        @test response.content == tracker.content
    end
    
    @testset "Response Schema (OpenAI GPT-5)" begin
        tracker = CallbackTracker()
        callback = create_tracked_callback(tracker)
        
        response = aigen(prompt, "openai:openai/gpt-5-mini"; streamcallback=callback)
        
        @test tracker.start_count == 1
        @test tracker.chunk_count >= 1
        @test tracker.done_count == 1
        @test tracker.meta_ai_count == 1  # Should be called with token info
        @test tracker.meta_usr_count == 0  # Response schema doesn't separate user tokens
        @test !isempty(tracker.content)
        @test response.content == tracker.content
        
        # Additional checks for Response schema
        @test response.tokens !== nothing
        if response.tokens !== nothing
            @test response.tokens.total_tokens > 0
        end
    end
    
    @testset "Token Counts Across Schemas" begin
        models = [
            "groq:moonshotai/kimi-k2-0905",
            "anthropic:anthropic/claude-haiku-4.5",
            "google-ai-studio:google/gemini-2.5-flash",
            "openai:openai/gpt-5-mini"
        ]
        
        for model in models
            @testset "Tokens for $model" begin
                tracker = CallbackTracker()
                callback = create_tracked_callback(tracker)
                
                response = aigen(prompt, model; streamcallback=callback)
                
                # Verify on_meta_ai was called (critical for all schemas)
                @test tracker.meta_ai_count >= 1
                
                # Verify final response has token information
                @test response.tokens !== nothing
                if response.tokens !== nothing
                    @test response.tokens.prompt_tokens >= 0
                    @test response.tokens.completion_tokens >= 0
                    @test response.tokens.total_tokens >= 0
                end
            end
        end
    end
end