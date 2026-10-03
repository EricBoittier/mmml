import setuptools
import shutil
import os

# Clean up stale build artifacts that cause "File exists" errors
# This fixes: [Errno 17] File exists: 'build/bdist.../wheel/pycharmm-X.X.X.dist-info'
build_dir = os.path.join(os.path.dirname(__file__), 'build')
if os.path.exists(build_dir):
    # Remove any existing .dist-info directories in the wheel build path
    bdist_wheel_dir = os.path.join(build_dir, 'bdist.linux-x86_64', 'wheel')
    if os.path.exists(bdist_wheel_dir):
        for item in os.listdir(bdist_wheel_dir):
            if item.endswith('.dist-info'):
                dist_info_path = os.path.join(bdist_wheel_dir, item)
                print(f"Cleaning stale dist-info: {dist_info_path}")
                shutil.rmtree(dist_info_path, ignore_errors=True)

setuptools.setup(
    packages=setuptools.find_packages()
)
