import os
import subprocess

# List of installed packages
packages = [
    "intel-opencl-icd",
    "intel-oneapi-runtime-dpcpp-cpp",
    "intel-oneapi-runtime-mkl",
    "intel-oneapi-common-vars",
    "intel-oneapi-common-oneapi-vars",
    "intel-oneapi-diagnostics-utility",
    "intel-oneapi-compiler-dpcpp-cpp",
    "intel-oneapi-dpcpp-ct",
    "intel-oneapi-mkl",
    "intel-oneapi-mkl-devel",
    "intel-oneapi-mpi",
    "intel-oneapi-mpi-devel",
    "intel-oneapi-dal",
    "intel-oneapi-dal-devel",
    "intel-oneapi-ippcp",
    "intel-oneapi-ippcp-devel",
    "intel-oneapi-ipp",
    "intel-oneapi-ipp-devel",
    "intel-oneapi-tlt",
    "intel-oneapi-ccl",
    "intel-oneapi-ccl-devel",
    "intel-oneapi-dnnl-devel",
    "intel-oneapi-dnnl",
    "intel-oneapi-tcm-1.0",
]

# Output file for results
output_file = "installed_packages_versions.txt"

# Check each package version and write to the output file
def get_package_version(package):
    try:
        result = subprocess.check_output(
            ["dpkg", "-l"], universal_newlines=True
        )
        for line in result.splitlines():
            if line.startswith("ii") and package in line:
                return line.split()[1] + "=" + line.split()[2]
        return None
    except subprocess.CalledProcessError:
        return None

def main():
    with open(output_file, "w") as f:
        f.write("Checking installed versions of specified packages...\n")
        install_line = "Use the following line to quickly install the needed packages. \nsudo apt install --simulate \\\n"
        for package in packages:
            version = get_package_version(package)
            if version:
                f.write(f"\t{package} --> {version}\n")
                install_line += f"    {version} \\\n"
            else:
                f.write(f"{package} --> not installed\n")
                install_line += f"    {package} \\\n"
        # Remove trailing backslash and newline
        install_line = install_line.rstrip("\\\n")
        f.write("\n" + install_line + "\n")

    print(f"Version information saved to {output_file}")
    with open(output_file, "r") as f:
        print(f.read())

if __name__ == "__main__":
    main()
