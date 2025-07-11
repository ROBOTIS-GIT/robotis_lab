# Robotis Lab Release Notes

0.1.1 (2025-07-11)
------------------
### Sim2Real Deployment Support Added
* Developed Sim2Real deployment pipeline for the OMY Reach task.
* Enabled running policies trained in Isaac Sim on real-world OMY robots.
* Provided detailed usage instructions and demonstration videos in the README.
* Refactored folder structure for source and scripts to improve maintainability.

0.1.0 (2025-07-01)
------------------
### Initial Release
* Developed as an external package for Isaac Lab
* Verified compatibility with the following environments:
    * Isaac Sim 4.5.0 and 5.0.0
    * Isaac Lab 2.1.0 and feature/isaacsim_5_0 branch
* Introduced simulation environments for reinforcement learning and imitation learning, featuring two Robotis robots:
    * OMY
    * FFW
* Enables users to conduct training and research using Robotis robots with Isaac Lab, including full support for custom tutorials in reinforcement and imitation learning scenarios.
