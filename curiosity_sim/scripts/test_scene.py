#!/usr/bin/env python3
# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure, or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
import sys
import time
import signal
import math

from isaacsim import SimulationApp

real_frame_per_second = 60.0
internal_frame_per_second = 60.0
time_steps_per_second = 100
kit = None
is_processing = False


def scheduler(signum, frame):
    global kit
    global is_processing
    if not is_processing:
        is_processing = True
        kit.update()
        is_processing = False

def get_root_prim():
    """Get the root prim of the stage."""
    import omni
    stage_handle = omni.usd.get_context().get_stage()
    root_prim = stage_handle.GetDefaultPrim()
    if not root_prim:
        root_prim = stage_handle.GetPseudoRoot().GetChildren()[0]
    return root_prim

def add_usd_to_stage(prim_path: str, root_prim_path: str, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
    import omni.isaac.core.utils.stage as stage_utils
    from pxr import UsdGeom

    prim_name = os.path.splitext(os.path.basename(prim_path))[0]
    stage_prim_path = f"{root_prim_path}/{prim_name}"

    obj = stage_utils.add_reference_to_stage(prim_path, stage_prim_path)
    if obj.IsValid():
        obj_xform = UsdGeom.Xformable(obj)
        xform_ops = obj_xform.GetOrderedXformOps()

        obj_xform.ClearXformOpOrder()

        translate_op = obj_xform.AddTranslateOp()
        translate_op.Set((x, y, z))

        rotate_op = obj_xform.AddRotateXYZOp()
        rotate_op.Set((roll * 180.0 / math.pi, pitch * 180.0 / math.pi, yaw * 180.0 / math.pi))

    return obj


def main():
    global kit

    robot_name = "curiosity_mars_rover"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    usd_path = os.path.join(script_dir, "..", "maps", "test_ground", "test_ground.usdc")
    curiosity_path = os.path.join(script_dir, "..", "models", "curiosity_mars_rover", "curiosity_mars_rover.usd")

    kit = SimulationApp({"renderer": "RayTracedLighting", "headless": False, "open_usd": usd_path})

    import omni
    from omni.isaac.core.utils.extensions import enable_extension
    from omni.isaac.core.utils.prims import get_articulation_root_api_prim_path
    from pxr import Sdf, Gf, UsdGeom, UsdPhysics, PhysxSchema
    import omni.graph.core as og
    from omni.graph.core import GraphPipelineStage

    enable_extension("omni.graph.action")
    kit.update()
    enable_extension("omni.isaac.ros2_bridge")
    kit.update()
    enable_extension("omni.isaac.repl")

    stage_handle = omni.usd.get_context().get_stage()

    # Setup physics
    scene = UsdPhysics.Scene.Define(stage_handle, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(3.71)
    PhysxSchema.PhysxSceneAPI.Apply(stage_handle.GetPrimAtPath("/physicsScene"))
    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage_handle, "/physicsScene")
    physxSceneAPI.CreateEnableCCDAttr(True)
    physxSceneAPI.CreateEnableStabilizationAttr(True)
    physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
    physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
    physxSceneAPI.CreateSolverTypeAttr("PGS")
    physxSceneAPI.CreateTimeStepsPerSecondAttr(time_steps_per_second)

    root_prim = get_root_prim()
    root_prim_path = root_prim.GetPath().pathString
    target_prim = f"{root_prim_path}/{robot_name}/base_footprint"

    # Add the Curiosity rover USD to the stage
    add_usd_to_stage(curiosity_path, root_prim_path, x=0.0, y=0.0, z=1.0, roll=0.0, pitch=0.0, yaw=0.0)

    graph_path = f"{root_prim_path}/{robot_name}/RobotControlGraph"

    # Delete the control graph if it already exists
    if stage_handle.GetPrimAtPath(graph_path):
        omni.kit.commands.execute("DeletePrims", paths=[graph_path])

    art_path = get_articulation_root_api_prim_path(target_prim)

    # Create new control graph
    (ros_control_graph, _, _, _) = og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPhysicsStep", "omni.isaac.core_nodes.OnPhysicsStep"),
                ("PublishJointState", "omni.isaac.ros2_bridge.ROS2Publisher"),
                ("ArticulationState", "omni.isaac.core_nodes.IsaacArticulationState"),
                ("SubscribeJointState", "omni.isaac.ros2_bridge.ROS2Subscriber"),
                ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
                ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("TimeSplitter", "omni.isaac.core_nodes.IsaacTimeSplitter"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPhysicsStep.outputs:step", "PublishJointState.inputs:execIn"),
                ("OnPhysicsStep.outputs:step", "ArticulationState.inputs:execIn"),
                ("OnPhysicsStep.outputs:step", "SubscribeJointState.inputs:execIn"),
                ("OnPhysicsStep.outputs:step", "ArticulationController.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "TimeSplitter.inputs:time"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ArticulationController.inputs:targetPrim", art_path),
                ("ArticulationState.inputs:targetPrim", art_path),
                ("PublishJointState.inputs:messageName", "JointState"),
                ("PublishJointState.inputs:messagePackage", "sensor_msgs"),
                ("PublishJointState.inputs:topicName", f"/{robot_name}/joint_states"),
                ("SubscribeJointState.inputs:messageName", "JointState"),
                ("SubscribeJointState.inputs:messagePackage", "sensor_msgs"),
                ("SubscribeJointState.inputs:topicName", f"/{robot_name}/joint_command"),
            ],
        },
    )

    og.Controller.connect(
        f"{graph_path}/SubscribeJointState.outputs:name", f"{graph_path}/ArticulationController.inputs:jointNames"
    )
    og.Controller.connect(
        f"{graph_path}/SubscribeJointState.outputs:position",
        f"{graph_path}/ArticulationController.inputs:positionCommand",
    )
    og.Controller.connect(
        f"{graph_path}/SubscribeJointState.outputs:velocity",
        f"{graph_path}/ArticulationController.inputs:velocityCommand",
    )
    og.Controller.connect(
        f"{graph_path}/SubscribeJointState.outputs:effort", f"{graph_path}/ArticulationController.inputs:effortCommand"
    )

    og.Controller.connect(
        f"{graph_path}/TimeSplitter.outputs:seconds", f"{graph_path}/PublishJointState.inputs:header:stamp:sec"
    )
    og.Controller.connect(
        f"{graph_path}/TimeSplitter.outputs:nanoseconds", f"{graph_path}/PublishJointState.inputs:header:stamp:nanosec"
    )
    og.Controller.connect(
        f"{graph_path}/ArticulationState.outputs:jointNames", f"{graph_path}/PublishJointState.inputs:name"
    )
    og.Controller.connect(
        f"{graph_path}/ArticulationState.outputs:jointPositions", f"{graph_path}/PublishJointState.inputs:position"
    )
    og.Controller.connect(
        f"{graph_path}/ArticulationState.outputs:jointVelocities", f"{graph_path}/PublishJointState.inputs:velocity"
    )
    og.Controller.connect(
        f"{graph_path}/ArticulationState.outputs:measuredJointEfforts", f"{graph_path}/PublishJointState.inputs:effort"
    )

    og.Controller.evaluate_sync(ros_control_graph)

    # Create Camera Graph
    camera_graph_path = f"{root_prim_path}/{robot_name}/CameraGraph"
    camera_prim_path = f"{root_prim_path}/{robot_name}/camera_link/Camera"

    # Delete the camera graph if it already exists
    if stage_handle.GetPrimAtPath(camera_graph_path):
        omni.kit.commands.execute("DeletePrims", paths=[camera_graph_path])

    # Define the camera prim
    camera_prim = UsdGeom.Camera.Define(stage_handle, Sdf.Path(camera_prim_path))
    xform_api = UsdGeom.XformCommonAPI(camera_prim)
    xform_api.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))
    xform_api.SetRotate(Gf.Vec3f(90, 0, -90), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    # Set the horizontal FOV
    resolution_height = 800
    resolution_width = 800

    # Set the focal length
    focal_length = 12.0  # mm
    camera_prim.GetFocalLengthAttr().Set(focal_length)

    # Create new camera graph
    (ros_camera_graph, _, _, _) = og.Controller.edit(
        {"graph_path": camera_graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("CreateRenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                ("RunOneSimulationFrame", "omni.isaac.core_nodes.OgnIsaacRunOneSimulationFrame"),
                ("ROS2Context", "omni.isaac.ros2_bridge.ROS2Context"),
                ("ROS2CameraHelper", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "RunOneSimulationFrame.inputs:execIn"),
                ("RunOneSimulationFrame.outputs:step", "CreateRenderProduct.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "ROS2CameraHelper.inputs:execIn"),
                ("CreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelper.inputs:renderProductPath"),
                ("ROS2Context.outputs:context", "ROS2CameraHelper.inputs:context"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("CreateRenderProduct.inputs:cameraPrim", camera_prim_path),
                ("CreateRenderProduct.inputs:enabled", True),
                ("CreateRenderProduct.inputs:height", resolution_height),
                ("CreateRenderProduct.inputs:width", resolution_width),
                ("ROS2CameraHelper.inputs:frameId", "camera_link"),
                ("ROS2CameraHelper.inputs:topicName", "/image_raw"),
                ("ROS2CameraHelper.inputs:type", "rgb"),
            ],
        },
    )

    og.Controller.evaluate_sync(ros_camera_graph)

    # Create Clock Graph
    clock_graph_path = f"{root_prim_path}/{robot_name}/ClockGraph"

    # Delete the clock graph if it already exists
    if stage_handle.GetPrimAtPath(clock_graph_path):
        omni.kit.commands.execute("DeletePrims", paths=[clock_graph_path])

    # Create new clock graph
    (ros_clock_graph, _, _, _) = og.Controller.edit(
        {"graph_path": clock_graph_path, "evaluator_name": "execution"},
        {
            # Define the nodes to be created within the graph
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ROS2Context", "omni.isaac.ros2_bridge.ROS2Context"),
                ("IsaacReadSimulationTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("ROS2PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
            ],
            # Define the connections between nodes
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ROS2PublishClock.inputs:execIn"),
                ("IsaacReadSimulationTime.outputs:simulationTime", "ROS2PublishClock.inputs:timeStamp"),
                ("ROS2Context.outputs:context", "ROS2PublishClock.inputs:context"),
            ],
            # Set the values for the ROS2PublishClock node
            og.Controller.Keys.SET_VALUES: [
                ("ROS2PublishClock.inputs:topicName", "/clock"),
            ],
        },
    )

    # Synchronously evaluate the graph to apply changes
    og.Controller.evaluate_sync(ros_clock_graph)

    # Create TF_Odometry Graph
    odometry_graph_path = f"{root_prim_path}/{robot_name}/OdometryGraph"

    # Delete the odom graph if it already exists
    if stage_handle.GetPrimAtPath(odometry_graph_path):
        omni.kit.commands.execute("DeletePrims", paths=[odometry_graph_path])

    # Create new odom graph
    (odometry_graph, _, _, _) = og.Controller.edit(
        {"graph_path": odometry_graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ROS2Context", "omni.isaac.ros2_bridge.ROS2Context"),
                ("IsaacReadSimulationTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("IsaacComputeOdometry", "omni.isaac.core_nodes.IsaacComputeOdometry"),
                ("ROS2PublishOdometry", "omni.isaac.ros2_bridge.ROS2PublishOdometry"),
                ("ROS2PublishRawTransformTree", "omni.isaac.ros2_bridge.ROS2PublishRawTransformTree"),
                ("ROS2PublishTransformTree", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                ("ROS2PublishStaticRawTransformTree", "omni.isaac.ros2_bridge.ROS2PublishRawTransformTree"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "IsaacComputeOdometry.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ROS2PublishRawTransformTree.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ROS2PublishTransformTree.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ROS2PublishStaticRawTransformTree.inputs:execIn"),
                ("IsaacReadSimulationTime.outputs:simulationTime", "ROS2PublishOdometry.inputs:timeStamp"),
                ("IsaacReadSimulationTime.outputs:simulationTime", "ROS2PublishRawTransformTree.inputs:timeStamp"),
                ("IsaacReadSimulationTime.outputs:simulationTime", "ROS2PublishTransformTree.inputs:timeStamp"),
                (
                    "IsaacReadSimulationTime.outputs:simulationTime",
                    "ROS2PublishStaticRawTransformTree.inputs:timeStamp",
                ),
                ("ROS2Context.outputs:context", "ROS2PublishOdometry.inputs:context"),
                ("ROS2Context.outputs:context", "ROS2PublishRawTransformTree.inputs:context"),
                ("ROS2Context.outputs:context", "ROS2PublishTransformTree.inputs:context"),
                ("ROS2Context.outputs:context", "ROS2PublishStaticRawTransformTree.inputs:context"),
                ("IsaacComputeOdometry.outputs:execOut", "ROS2PublishOdometry.inputs:execIn"),
                ("IsaacComputeOdometry.outputs:linearVelocity", "ROS2PublishOdometry.inputs:linearVelocity"),
                ("IsaacComputeOdometry.outputs:angularVelocity", "ROS2PublishOdometry.inputs:angularVelocity"),
                ("IsaacComputeOdometry.outputs:position", "ROS2PublishOdometry.inputs:position"),
                ("IsaacComputeOdometry.outputs:orientation", "ROS2PublishOdometry.inputs:orientation"),
                ("IsaacComputeOdometry.outputs:position", "ROS2PublishRawTransformTree.inputs:translation"),
                ("IsaacComputeOdometry.outputs:orientation", "ROS2PublishRawTransformTree.inputs:rotation"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("IsaacComputeOdometry.inputs:chassisPrim", f"{root_prim_path}/{robot_name}/base_footprint"),
                ("ROS2PublishOdometry.inputs:chassisFrameId", "base_footprint"),
                ("ROS2PublishOdometry.inputs:topicName", "/model/curiosity_mars_rover/odometry"),
                ("ROS2PublishRawTransformTree.inputs:parentFrameId", "odom"),
                ("ROS2PublishRawTransformTree.inputs:childFrameId", "base_footprint"),
                ("ROS2PublishStaticRawTransformTree.inputs:parentFrameId", "world"),
                ("ROS2PublishStaticRawTransformTree.inputs:childFrameId", "odom"),
                ("ROS2PublishTransformTree.inputs:parentPrim", f"{root_prim_path}/{robot_name}/base_footprint"),
                ("ROS2PublishTransformTree.inputs:targetPrims", f"{root_prim_path}/{robot_name}/base_footprint"),
            ],
        },
    )

    # Synchronously evaluate the graph to apply changes
    og.Controller.evaluate_sync(odometry_graph)

    # Define the path for the Lidar Graph
    lidar_graph_path = f"{root_prim_path}/{robot_name}/LidarGraph"
    lidar_prim_path = f"{root_prim_path}/{robot_name}/lidar_link/Lidar"

    # Delete the Lidar graph if it already exists
    if stage_handle.GetPrimAtPath(lidar_graph_path):
        omni.kit.commands.execute("DeletePrims", paths=[lidar_graph_path])

    # Create the Lidar prim
    _, lidar_sensor = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path=lidar_prim_path,
        parent=None,
        config='RPLIDAR_S2E',
        translation=(0, 0, 0),
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
    )

    # Create a new Lidar graph
    (ros_lidar_graph, _, _, _) = og.Controller.edit(
        {"graph_path": lidar_graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("CreateRenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                ("RunOneSimulationFrame", "omni.isaac.core_nodes.OgnIsaacRunOneSimulationFrame"),
                ("ROS2Context", "omni.isaac.ros2_bridge.ROS2Context"),
                ("ROS2RtxLidarHelper", "omni.isaac.ros2_bridge.ROS2RtxLidarHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "RunOneSimulationFrame.inputs:execIn"),
                ("RunOneSimulationFrame.outputs:step", "CreateRenderProduct.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "ROS2RtxLidarHelper.inputs:execIn"),
                ("CreateRenderProduct.outputs:renderProductPath", "ROS2RtxLidarHelper.inputs:renderProductPath"),
                ("ROS2Context.outputs:context", "ROS2RtxLidarHelper.inputs:context"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("CreateRenderProduct.inputs:cameraPrim", lidar_prim_path),
                ("CreateRenderProduct.inputs:enabled", True),
                ("ROS2RtxLidarHelper.inputs:frameId", "lidar_link"),
                ("ROS2RtxLidarHelper.inputs:topicName", "/scan"),
                ("ROS2RtxLidarHelper.inputs:type", "laser_scan"),
            ],
        },
    )

    # Synchronously evaluate the graph to apply changes
    og.Controller.evaluate_sync(ros_lidar_graph)

    signal.signal(signal.SIGALRM, scheduler)
    signal.setitimer(signal.ITIMER_REAL, 1 / real_frame_per_second, 1 / real_frame_per_second)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        kit.close()


if __name__ == "__main__":
    main()
