
# List of CUDA versions supported by RAPIDSAI
const supported_versions = ["11.2", "11.8", "12.0"]

function find_closest_supported_version(major, supported_versions)
    major_versions = filter(v -> startswith(v, major), supported_versions)

    if !isempty(major_versions)
        return maximum(major_versions)
    end
    less_than_major = filter(v -> parse(Int, split(v, ".")[1]) < parse(Int, major),
                             supported_versions)

    return !isempty(less_than_major) ? maximum(less_than_major) : nothing
end

function set_conda_cuda_version!()
    cuda_version = CUDA.driver_version()
    major, _, _ = split("$cuda_version", ".")
    @assert major in ["11", "12"] "CUDA Version $cuda_version is not supported."
    closest_version = find_closest_supported_version(major, supported_versions)

    dfile = replace(dirname(pathof(RAPIDS)), "src" => "CondaPkg.toml")
    cur_deps = CondaPkg.read_deps(; file=dfile)["deps"]

    if haskey(cur_deps, "cuda-version")
        if cur_deps["cuda-version"] != "=$closest_version"
            CondaPkg.add("cuda-version"; version="=$closest_version", channel="nvidia")
        end
    else
        CondaPkg.add("cuda-version"; version="=$closest_version", channel="nvidia")
    end
end
