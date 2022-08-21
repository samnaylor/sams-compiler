from subprocess import CalledProcessError, check_output

expected_results = {
    "factorial_main": 153,
    "fibonacci": 144
}

all_passed = True

for key, value in expected_results.items():
    try:
        check_output([f"./bin/{key}"])
        check_output(["echo", "$?"])
    except CalledProcessError as cpe:
        if not value == cpe.returncode:
            all_passed = False
            print(f"Test '{key}' Failed. Expected {value} Got {cpe.returncode}.")

if all_passed:
    print("All Tests Passed!")
