# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Seongwoo Kim

"""Install the Cyclo Arena extension package."""

from pathlib import Path
from tomllib import load

from setuptools import find_packages, setup

PACKAGE_ROOT = Path(__file__).resolve().parent
with (PACKAGE_ROOT / "config" / "extension.toml").open("rb") as metadata_file:
    EXTENSION_METADATA = load(metadata_file)
PROFILE_FILES = [
    str(path.relative_to(PACKAGE_ROOT)) for path in sorted((PACKAGE_ROOT / "configs" / "profiles").glob("*.yaml"))
]

setup(
    name="cyclo_arena",
    version=EXTENSION_METADATA["package"]["version"],
    description=EXTENSION_METADATA["package"]["description"],
    author="ROBOTIS CO., LTD.",
    maintainer="ROBOTIS CO., LTD.",
    url=EXTENSION_METADATA["package"]["repository"],
    packages=find_packages(),
    include_package_data=True,
    data_files=[("share/cyclo_arena/profiles", PROFILE_FILES)],
    python_requires=">=3.11",
    install_requires=["cyclo_lab"],
    entry_points={"console_scripts": ["cyclo-arena=cyclo_arena.cli:main"]},
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
    ],
    zip_safe=False,
)
