using JSON3
using Serialization
using CodecZlib
using StructTypes
using Random
using JLD2
using Chairmarks

# Define a struct to represent model information
struct ModelInfo
    id::String
    name::String
    description::String
    context_length::Int
    pricing::Dict{String, Float64}
    created::Int
    owned_by::String
    parameters::Int
    top_provider::Dict{String, Union{String, Int, Bool, Float64}}
end

# Make it compatible with StructTypes for efficient JSON parsing
StructTypes.StructType(::Type{ModelInfo}) = StructTypes.Struct()

# Generate dummy data simulating a large curl response with 1000 models
function generate_dummy_data(n_models::Int = 1000)
    Random.seed!(42)  # For reproducible results
    
    models = ModelInfo[]
    
    for i in 1:n_models
        model = ModelInfo(
            "model-$(i)-$(randstring(8))",
            "Model $(i) $(randstring(6))",
            "This is a description for model $(i). " * randstring(100),
            rand([4096, 8192, 16384, 32768, 128000]),
            Dict(
                "prompt" => rand() * 0.01,
                "completion" => rand() * 0.02
            ),
            1600000000 + rand(1:100000000),
            "provider-$(rand(1:20))",
            rand([7, 13, 70, 175, 405]) * 1_000_000_000,
            Dict(
                "name" => "provider-$(rand(1:10))",
                "max_completion_tokens" => rand([4096, 8192, 16384]),
                "is_moderated" => rand(Bool)
            )
        )
        push!(models, model)
    end
    
    return models
end

# Benchmark functions
function benchmark_json_write_read(data, filepath)
    # Write
    write_time = @elapsed begin
        open(filepath, "w") do io
            JSON3.write(io, data)
        end
    end
    
    # Read
    read_time = @elapsed begin
        result = open(filepath, "r") do io
            JSON3.read(io)
        end
    end
    
    return write_time, read_time
end

function benchmark_gzip_json_write_read(data, filepath)
    # Write
    write_time = @elapsed begin
        open(filepath, "w") do io
            stream = GzipCompressorStream(io)
            JSON3.write(stream, data)
            close(stream)
        end
    end
    
    # Read  
    read_time = @elapsed begin
        result = open(filepath, "r") do io
            stream = GzipDecompressorStream(io)
            data = JSON3.read(stream)
            close(stream)
            data
        end
    end
    
    return write_time, read_time
end

function benchmark_binary_serialize_write_read(data, filepath)
    # Write
    write_time = @elapsed begin
        serialize(filepath, data)
    end
    
    # Read
    read_time = @elapsed begin
        result = deserialize(filepath)
    end
    
    return write_time, read_time
end

function benchmark_structtypes_json3_write_read(data, filepath)
    # Write (same as regular JSON3 but optimized for reading)
    write_time = @elapsed begin
        open(filepath, "w") do io
            JSON3.write(io, data)
        end
    end
    
    # Read with StructTypes optimization
    read_time = @elapsed begin
        result = open(filepath, "r") do io
            JSON3.read(io, Vector{ModelInfo})
        end
    end
    
    return write_time, read_time
end

function benchmark_jld2_write_read(data, filepath)
    # Write
    write_time = @elapsed begin
        jldsave(filepath; data=data)
    end
    
    # Read
    read_time = @elapsed begin
        result = load(filepath, "data")
    end
    
    return write_time, read_time
end

function get_file_size(filepath)
    return filesize(filepath)
end

function run_benchmarks()
    println("🚀 Starting I/O Format Benchmarks")
    println("=" ^ 50)
    
    # Generate test data
    println("📊 Generating dummy data (1000 models)...")
    data = generate_dummy_data(1000)
    println("✅ Data generated")
    
    # Create temp directory
    temp_dir = mktempdir()
    println("📁 Using temp directory: $temp_dir")
    
    # File paths
    json_file = joinpath(temp_dir, "data.json")
    gzip_file = joinpath(temp_dir, "data.json.gz")
    binary_file = joinpath(temp_dir, "data.bin")
    struct_file = joinpath(temp_dir, "data_struct.json")
    jld2_file = joinpath(temp_dir, "data.jld2")
    
    results = Dict()
    
    # 1. Regular JSON
    println("\n1️⃣  Testing Regular JSON...")
    write_time, read_time = benchmark_json_write_read(data, json_file)
    file_size = get_file_size(json_file)
    results["JSON"] = (
        write_time = write_time,
        read_time = read_time,
        total_time = write_time + read_time,
        file_size = file_size
    )
    
    # 2. Gzipped JSON
    println("2️⃣  Testing Gzipped JSON...")
    write_time, read_time = benchmark_gzip_json_write_read(data, gzip_file)
    file_size = get_file_size(gzip_file)
    results["Gzipped JSON"] = (
        write_time = write_time,
        read_time = read_time,
        total_time = write_time + read_time,
        file_size = file_size
    )
    
    # 3. Binary Serialization
    println("3️⃣  Testing Binary Serialization...")
    write_time, read_time = benchmark_binary_serialize_write_read(data, binary_file)
    file_size = get_file_size(binary_file)
    results["Binary"] = (
        write_time = write_time,
        read_time = read_time,
        total_time = write_time + read_time,
        file_size = file_size
    )
    
    # 4. JLD2
    println("4️⃣  Testing JLD2...")
    write_time, read_time = benchmark_jld2_write_read(data, jld2_file)
    file_size = get_file_size(jld2_file)
    results["JLD2"] = (
        write_time = write_time,
        read_time = read_time,
        total_time = write_time + read_time,
        file_size = file_size
    )
    
    # 5. StructTypes + JSON3
    println("5️⃣  Testing StructTypes + JSON3...")
    write_time, read_time = benchmark_structtypes_json3_write_read(data, struct_file)
    file_size = get_file_size(struct_file)
    results["StructTypes+JSON3"] = (
        write_time = write_time,
        read_time = read_time,
        total_time = write_time + read_time,
        file_size = file_size
    )
    
    # Display results
    println("\n" * "=" ^ 80)
    println("📈 BENCHMARK RESULTS")
    println("=" ^ 80)
    
    # Sort by total time
    sorted_results = sort(collect(results), by = x -> x[2].total_time)
    
    println("Format                | Write (s) | Read (s)  | Total (s) | Size (MB) | Compression")
    println("-" ^ 80)
    
    json_size = results["JSON"].file_size
    
    for (format, result) in sorted_results
        compression_ratio = json_size / result.file_size
        println("$(rpad(format, 20)) | $(lpad(round(result.write_time, digits=4), 8)) | $(lpad(round(result.read_time, digits=4), 8)) | $(lpad(round(result.total_time, digits=4), 8)) | $(lpad(round(result.file_size/1024/1024, digits=2), 8)) | $(lpad(round(compression_ratio, digits=2), 6))x")
    end
    
    println("\n🏆 Winner: $(sorted_results[1][1]) (fastest total time)")
    
    # Detailed benchmark with Chairmarks
    println("\n" * "=" ^ 80)
    println("🔬 DETAILED BENCHMARKS (using Chairmarks)")
    println("=" ^ 80)
    
    println("\n📝 JSON Write:")
    json_write_bench = @b benchmark_json_write_read(data, json_file)[1]
    display(json_write_bench)
    
    println("\n📖 JSON Read:")
    benchmark_json_write_read(data, json_file)  # Ensure file exists
    json_read_bench = @b benchmark_json_write_read(data, json_file)[2]
    display(json_read_bench)
    
    println("\n📝 Gzipped JSON Write:")
    gzip_write_bench = @b benchmark_gzip_json_write_read(data, gzip_file)[1]
    display(gzip_write_bench)
    
    println("\n📖 Gzipped JSON Read:")
    benchmark_gzip_json_write_read(data, gzip_file)  # Ensure file exists
    gzip_read_bench = @b benchmark_gzip_json_write_read(data, gzip_file)[2]
    display(gzip_read_bench)
    
    println("\n📝 Binary Write:")
    binary_write_bench = @b benchmark_binary_serialize_write_read(data, binary_file)[1]
    display(binary_write_bench)
    
    println("\n📖 Binary Read:")
    benchmark_binary_serialize_write_read(data, binary_file)  # Ensure file exists
    binary_read_bench = @b benchmark_binary_serialize_write_read(data, binary_file)[2]
    display(binary_read_bench)
    
    println("\n📝 JLD2 Write:")
    jld2_write_bench = @b benchmark_jld2_write_read(data, jld2_file)[1]
    display(jld2_write_bench)
    
    println("\n📖 JLD2 Read:")
    benchmark_jld2_write_read(data, jld2_file)  # Ensure file exists
    jld2_read_bench = @b benchmark_jld2_write_read(data, jld2_file)[2]
    display(jld2_read_bench)
    
    # Cleanup
    rm(temp_dir, recursive=true)
    println("\n🧹 Cleanup completed")
    
    return results
end

run_benchmarks()