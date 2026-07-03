# FIXTURE: excerpt of warplab/isaac-auv-env warpauv_env.py @ 7c5ebe7f7a08acd2570b5fba328e92b7f59f6794
# Contains every content anchor used by apply_bluerov2_patch.py / training.py.
# Elided regions are replaced by _fixture_seam_N methods so the excerpt stays
# valid Python (the patch script py_compiles what it edits). Anchor lines are
# byte-identical to the pristine reference. Regenerate from the reference clone.
##
from isaaclab.utils.math import quat_apply, quat_conjugate
from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .thruster_dynamics import DynamicsFirstOrder, ConversionFunctionBasic, get_thruster_com_and_orientations

class WarpAUVEnvWindow(BaseEnvWindow):
    """Window manager for the warpauvenv environment."""

    def __init__(self, env: WarpAUVEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class WarpAUVEnvCfg(DirectRLEnvCfg):
    ui_window_class_type = WarpAUVEnvWindow

    sim: SimulationCfg = SimulationCfg(dt=1 / 120)

    # robot
    robot_cfg: RigidObjectCfg = WARPAUV_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)
    debug_vis = True

    observation_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
    action_space: gym.spaces.Space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float64)
    state_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
    # env
    decimation = 2
    cap_episode_length = True
    episode_length_s = 3.0
    episode_length_before_reset = None
    num_actions = 6
    num_observations = 17
    num_states = 0
    use_boundaries = True
    max_auv_x = 7
    max_auv_y = 7
    max_auv_z = 7
    starting_depth = 8
    min_goal_steps = 100
    goal_completion_radius = 0.01
    goal_dims = 4
    eval_mode = False

    goal_spawn_radius = 2.0
    init_guidance_rate = 0.1
    init_vel_max = 1.0

    # rewards
    rew_scale_terminated = 0.0
    rew_scale_alive = 0.0
    rew_scale_completion = 1000

    rew_scale_pos = 0.2
    rew_scale_ang = 0.5
    rew_scale_vel = 0.0
    rew_scale_ang_vel = 0.0
    rew_scale_lin_vel = 0.0
    rew_scale_actions = 0.2

    # dynamics
    com_to_cob_offset = [0.0, 0.0, 0.01] # in meters, add this (xyz) to COM to get COB location
    water_rho = 997.0 # kg/m^3
    water_beta = 0.001306 # Pa s, dynamic viscosity of water @ 50 deg F
    rotor_constant = 0.1 / 100.0 # rotor constant used in Gazebo, note /10 because 0.04 is "10x bigger than it should be"
    dyn_time_constant = 0.05 # time constant for linear dynamics for each rotor 
    volume = 0.022747843530591776 # assuming cubic meters - NEUTRALLY BOUYANT. In orignal sim file volume = 0.0223
    # volume = 0.03
    mass = 2.2701e+01 # kg

     # domain randomization
    # todo: isaaclabs has a built-in method somehow
    class domain_randomization:
        use_custom_randomization = True
        # com_to_cob_offset_radius = 0 # uniform from sphere around predicted com_to_cob_offset
        # volume_range = [0.022747843530591776, 0.022747843530591776] # uniform [lowerbound, upperbound]
        # mass_range = [2.2701e+0,2.2701e+0] # uniform [lowerbound, upperbound]
        com_to_cob_offset_radius = 0.05 # uniform from sphere around predicted com_to_cob_offset
        volume_range = [0.019747843530591773, 0.02574784353059178] # uniform [loierbound, upperbound]
        mass_range = [2.2701e+01,2.2701e+01] # uniform [lowerbound, upperbound]


class WarpAUVEnv(DirectRLEnv):
    cfg: WarpAUVEnvCfg

    def __init__(self, cfg: WarpAUVEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Debug mode?
        self._debug = False

        # Initialize buffers
        self._actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._goal = torch.zeros(self.num_envs, self.cfg.goal_dims, device=self.device)
        self._default_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._completion_buffer = torch.zeros(self.num_envs, device=self.device)
        self._completed_envs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    # [... elided ...]
    def _fixture_seam_1(self, actions=None):

        # Get specific information about the AUV
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        # todo: get inertias from the model or physx view
        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float, requires_grad=False)

        # estimated inertial values from a solid rect. prism model (with estimated side lengths of 0.7m, 0.4m, and 0.2m):
        # fake inertial values for warpauv, based on I_ii = (1/12) * mass * (len_j**2 + len_k**2)
        self.inertia_tensors[:, 0] = 0.37
        self.inertia_tensors[:, 1] = 0.97
        self.inertia_tensors[:, 2] = 1.19

        if self.cfg.mass:
            self.masses = torch.full((self.num_envs, 1), self.cfg.mass, device=self.device)
        else:
            self.masses = self._robot.root_physx_view._masses

        # todo: cleaner way to handle this
        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
            self.com_to_cob_offsets = torch.tensor(self.cfg.com_to_cob_offset).repeat(self.num_envs, 1).to(self.device)
        else:
            self.com_to_cob_offsets = self.cfg.com_to_cob_offset.copy()

        if type(self.cfg.volume) != torch.Tensor:
            self.volumes = torch.full((self.num_envs, 1), self.cfg.volume, device=self.device)
        else:
            self.volumes = self.cfg.volume.copy()

        self.inertia_tensors_mean = self.inertia_tensors.mean(dim=1, keepdim=True) 

        # Initialize dynamics calculators
        self._init_thruster_dynamics()
        
        # Set initial goals
        self._reset_idx(self._robot._ALL_INDICES)


    def _init_thruster_dynamics(self):
        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
          self.cfg.com_to_cob_offset = torch.tensor(self.cfg.com_to_cob_offset, device=self.device, dtype=torch.float32, requires_grad=False).reshape(1,3).repeat(self.num_envs, 1)

        # get force calculation functions and rotor dynamics models
        self.force_calculation_functions = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.thruster_dynamics = DynamicsFirstOrder(self.num_envs, 6, self.cfg.dyn_time_constant, self.device)
        self.thruster_conversion = ConversionFunctionBasic(self.cfg.rotor_constant)

    def _setup_scene(self):
        self.cfg.robot_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.starting_depth))
        self._robot = RigidObject(self.cfg.robot_cfg)


    # [... elided ...]
    def _compute_dynamics(self, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute dynamics from actions.
        actions are -1 for full reverse thrust, 1 for full forward thrust - THESE REPRESENT PWM VALUES
        BASED ON LINE 91 of https://gitlab.com/warplab/ros/warpauv/warpauv_simulation/-/blob/master/src/robot_sim_interface.py
        Args:
            actions (torch.Tensor): Actions shape (num_envs, num_actions)

        Returns:
            [torch.Tensor]: Forces sent to the simulation
            [torch.Tensor]: Torques sent to the simulation
        """

        if self._debug: print("actions: ", actions)

        thruster_forces = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        thruster_torques = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        motorValues = torch.clone(actions) # at this point these are PWM commands between -1 and 1

        if self._debug: print("motorValues: ", motorValues)
        return thruster_forces, thruster_torques
