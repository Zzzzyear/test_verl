# 文件路径: src/egpo/main_egpo.py
import os
import sys
import socket
import hydra
import ray
from omegaconf import OmegaConf

# 基础组件导入
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.config import validate_config
from verl.utils.fs import copy_to_local
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn, RLHFDataset
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import Role

# EGPO 组件
from egpo.trainer.egpo_trainer import EGPOTrainer
from egpo.core_config import EGPOConfig

# Driver 端注册 (可选，防止 IDE 报错)
try:
    import egpo.signals.hybrid_reward_loop 
    import egpo.signals.hybrid_reward_manager 
except ImportError:
    pass

# =========================================================================
# 【核心修复】自定义 Worker 类 - 运行时拦截版
# =========================================================================
@ray.remote
class EGPOActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化时尝试修复一次
        self._fix_batch_size()

    def _fix_batch_size(self):
        """Helper to force set micro_batch_size on the underlying engine"""
        if not hasattr(self, 'actor') or self.actor is None:
            return
            
        engine = getattr(self.actor, 'engine', None)
        if engine:
            fixed = False
            # 检查常见属性名
            for attr in ['micro_batch_size', 'micro_batch_size_per_gpu']:
                current_val = getattr(engine, attr, None)
                if current_val is None:
                    setattr(engine, attr, 4) # 强制设为 4
                    fixed = True
            
            if fixed:
                print(f"[EGPO/Worker] 🔧 Runtime Fix: Forced Actor micro_batch_size to 4")

    def compute_log_prob(self, data):
        """
        拦截 compute_log_prob 调用，确保在执行 split 前参数已就位。
        """
        # 1. 再次执行修复 (防止被重置)
        self._fix_batch_size()
        
        # 2. 调用父类方法
        return super().compute_log_prob(data)

# =========================================================================

@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_egpo(config)

def run_egpo(config):
    if not ray.is_initialized():
        # 构造 runtime_env
        project_python_path = os.environ.get("PYTHONPATH", "")
        current_cwd = os.getcwd()
        
        env_vars = {
            "PYTHONPATH": project_python_path + ":" + current_cwd,
            "VLLM_USE_V1": "1",
            "RAY_DEDUP_LOGS": "0" 
        }
        
        default_runtime_env = get_ppo_ray_runtime_env()
        if hasattr(default_runtime_env, 'to_container'):
             default_runtime_env = OmegaConf.to_container(default_runtime_env, resolve=True)

        combined_env_vars = default_runtime_env.get("env_vars", {})
        combined_env_vars.update(env_vars)
        default_runtime_env["env_vars"] = combined_env_vars
        
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        if OmegaConf.is_config(ray_init_kwargs):
            ray_init_kwargs = OmegaConf.to_container(ray_init_kwargs, resolve=True)
        
        ray_init_kwargs["runtime_env"] = default_runtime_env
        
        print(f"[EGPO] Initializing Ray with PYTHONPATH: {env_vars['PYTHONPATH']}")
        ray.init(**ray_init_kwargs)

    # 注入 EGPO 默认配置
    if 'egpo' not in config:
        egpo_conf = OmegaConf.structured(EGPOConfig)
        OmegaConf.set_struct(config, False)
        config.egpo = egpo_conf
        OmegaConf.set_struct(config, True)

    runner = ray.remote(num_cpus=1)(TaskRunner).remote()
    ray.get(runner.run.remote(config))


class TaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        # 【关键】使用自定义 Worker 类
        actor_rollout_cls = EGPOActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
        
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
            
        # 注意：这里直接赋值类，不要 ray.remote()
        self.role_worker_mapping[role] = actor_rollout_cls
        self.mapping[role] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        if config.algorithm.adv_estimator in ['grpo', 'egpo_grpo']:
            print("[EGPO] Running GRPO, skipping Critic initialization.")
            return

        from verl.workers.fsdp_workers import CriticWorker
        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        self.mapping[Role.Critic] = "global_pool"

    def add_reward_model_worker(self, config):
        if config.reward_model.enable:
            from verl.workers.fsdp_workers import RewardModelWorker
            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            self.mapping[Role.RewardModel] = "global_pool"

    def init_resource_pool_mgr(self, config):
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def run(self, config):
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        
        try:
            import egpo.signals.hybrid_reward_manager
            import egpo.signals.hybrid_reward_loop
            print("[EGPO/TaskRunner] Modules registered.")
        except ImportError as e:
            print(f"[EGPO/TaskRunner] Registration Warning: {e}")

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)

        try:
            from verl.trainer.ppo.utils import need_critic, need_reference_policy
            use_ref = need_reference_policy(self.role_worker_mapping)
            is_grpo = config.algorithm.adv_estimator in ['grpo', 'egpo_grpo']
            use_critic_ = False if is_grpo else need_critic(config)
        except ImportError:
            use_ref = True 
            use_critic_ = config.algorithm.adv_estimator == 'gae'

        validate_config(config=config, use_reference_policy=use_ref, use_critic=use_critic_)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, 
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = self.create_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = self.create_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        
        from torch.utils.data import RandomSampler, SequentialSampler
        if config.data.shuffle:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = SequentialSampler(train_dataset)

        print("🚀 Initializing EGPOTrainer...")
        trainer = EGPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        
        trainer.init_workers()
        trainer.fit()

    def create_dataset(self, data_paths, data_config, tokenizer, processor, is_train=True):
        return RLHFDataset(
            data_files=data_paths,
            tokenizer=tokenizer,
            processor=processor,
            config=data_config,
            max_samples=data_config.get("train_max_samples", -1) if is_train else data_config.get("val_max_samples", -1)
        )

if __name__ == "__main__":
    main()