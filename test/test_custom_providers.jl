using Test
using OpenRouter
using OpenRouter: add_provider, remove_provider, add_model, remove_model, list_providers

@testset "Custom Providers and Models" begin
    
    @testset "Provider Management" begin
        # Get initial provider count
        initial_count = length(list_providers())
        
        # Add a custom provider
        add_provider("test-echo", "http://localhost:8080/v1", "Bearer", "TEST_API_KEY", 
                    Dict{String,String}(), "Test echo server")
        
        # Check it was added
        @test length(list_providers()) == initial_count + 1
        @test "test-echo" in list_providers()
        
        # Add another provider with different auth
        add_provider("test-custom", "http://localhost:9000/api", "x-api-key", nothing,
                    Dict("Custom-Header" => "test-value"), "Custom test server")
        
        @test length(list_providers()) == initial_count + 2
        @test "test-custom" in list_providers()
        
        # Remove providers
        remove_provider("test-echo")
        remove_provider("test-custom")
        
        @test length(list_providers()) == initial_count
        @test "test-echo" ∉ list_providers()
        @test "test-custom" ∉ list_providers()
    end
    
    @testset "Model Management" begin
        # Get initial model count
        initial_models = list_cached_models()
        initial_count = length(initial_models)
        
        # Add custom models
        cached_model1 = add_model("test/model1", "Test Model 1", "First test model", 4096)
        cached_model2 = add_model("test/model2", "Test Model 2", "Second test model", 8192)
        
        @test cached_model1 isa OpenRouter.CachedModel
        @test cached_model1.model.id == "test/model1"
        @test cached_model1.model.name == "Test Model 1"
        @test cached_model1.model.context_length == 4096
        
        # Check models were added to cache
        updated_models = list_cached_models()
        @test length(updated_models) == initial_count + 2
        
        # Test model retrieval
        retrieved = get_model("test/model1")
        @test retrieved !== nothing
        @test retrieved.model.id == "test/model1"
        @test retrieved.model.name == "Test Model 1"
        
        # Test search functionality
        test_models = search_models("test")
        @test length(test_models) >= 2
        test_model_ids = [cached.model.id for cached in test_models]
        @test "test/model1" in test_model_ids
        @test "test/model2" in test_model_ids
        
        # Test case-sensitive search
        test_models_case = search_models("TEST", case_sensitive=true)
        @test length(test_models_case) == 0  # Should find nothing with wrong case
        
        test_models_insensitive = search_models("TEST", case_sensitive=false)
        @test length(test_models_insensitive) >= 2  # Should find our test models
        
        # Remove models
        @test remove_model("test/model1") == true
        @test remove_model("test/model2") == true
        @test remove_model("nonexistent/model") == false
        
        # Check models were removed
        final_models = list_cached_models()
        @test length(final_models) == initial_count
        
        @test get_model("test/model1") === nothing
        @test get_model("test/model2") === nothing
    end
    
    @testset "Model with Custom Pricing and Architecture" begin
        # Create custom pricing
        pricing = OpenRouter.Pricing("0.001", "0.002", "0", "0", "0", "0", 
                                    nothing, nothing, nothing, "0", nothing, nothing)
        
        # Create custom architecture  
        arch = OpenRouter.Architecture("text->text", ["text"], ["text"], "custom-tokenizer", "chat")
        
        # Add model with custom pricing and architecture
        cached_model = add_model("custom/advanced", "Advanced Custom Model", 
                               "Model with custom pricing", 16384, pricing, arch)
        
        @test cached_model.model.pricing.prompt == "0.001"
        @test cached_model.model.pricing.completion == "0.002"
        @test cached_model.model.architecture.tokenizer == "custom-tokenizer"
        @test cached_model.model.architecture.instruct_type == "chat"
        @test cached_model.model.context_length == 16384
        
        # Cleanup
        remove_model("custom/advanced")
    end
    
    @testset "Provider Info Retrieval" begin
        # Add a test provider
        add_provider("test-info", "http://test.local/v1", "Bearer", "TEST_KEY",
                    Dict("X-Custom" => "value"), "Test provider info")
        
        # Test provider info retrieval
        info = OpenRouter.get_provider_info("test-info")
        @test info !== nothing
        @test info.base_url == "http://test.local/v1"
        @test info.auth_header_format == "Bearer"
        @test info.api_key_env_var == "TEST_KEY"
        @test info.default_headers["X-Custom"] == "value"
        @test info.notes == "Test provider info"
        
        # Test auth header building
        auth_header = OpenRouter.get_provider_auth_header("test-info", "secret123")
        @test auth_header !== nothing
        @test auth_header.first == "Authorization"
        @test auth_header.second == "Bearer secret123"
        
        # Cleanup
        remove_provider("test-info")
    end
    
    @testset "Integration with Existing Functionality" begin
        # Add a custom model
        add_model("integration/test", "Integration Test Model", "For integration testing", 2048)
        
        # Test that it appears in searches
        all_models = list_cached_models()
        integration_models = filter(m -> startswith(m.id, "integration/"), all_models)
        @test length(integration_models) == 1
        @test integration_models[1].id == "integration/test"
        
        # Test search functionality
        found_models = search_models("integration")
        @test length(found_models) >= 1
        @test any(cached -> cached.model.id == "integration/test", found_models)
        
        # Cleanup
        remove_model("integration/test")
    end
end