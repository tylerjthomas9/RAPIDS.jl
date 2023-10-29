
# List of CUDA versions supported by RAPIDSAI
const supported_versions = ["11.2", "11.8", "12.0"]

# Create a function to find the closest supported version
function find_closest_supported_version(major, supported_versions)
    # Filter versions with the same major number
    major_versions = filter(v -> startswith(v, major), supported_versions)

    # If there are versions with the same major number, choose the maximum one
    if length(major_versions) > 0
        return maximum(major_versions)
    else
        # Otherwise, choose the maximum supported version that is less than the system's major version
        less_than_major = filter(v -> parse(Int, split(v, ".")[1]) < parse(Int, major),
                                 supported_versions)
        if length(less_than_major) > 0
            return maximum(less_than_major)
        else
            return nothing  # No compatible version found
        end
    end
    return error("No compatible CUDA version found for CUDA $(CUDA.driver_version())")
end

function set_conda_cuda_version!()
    major, minor, patch = split("$(CUDA.driver_version())", ".")
    closest_version = find_closest_supported_version(major, supported_versions)
    return CondaPkg.add("cuda-version"; version="=$closest_version")
end
