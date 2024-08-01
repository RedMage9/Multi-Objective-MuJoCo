from gym.envs.registration import register


register(
    id='MOHalfCheetah-v0',
    entry_point='mo_envs.half_cheetah:HalfCheetahEnv',
    max_episode_steps=1000,
)

register(
    id='MOAnt-v0',
    entry_point='mo_envs.ant:AntEnv',
    max_episode_steps=1000,
)

register(
    id='MOWalker2d-v0',
    entry_point='mo_envs.walker2d:Walker2dEnv',
    max_episode_steps=1000,
)

register(
    id='MOHopper-v0',
    entry_point='mo_envs.hopper:HopperEnv',
    max_episode_steps=1000,
)

register(
    id='MOHumanoid-v0',
    entry_point='mo_envs.humanoid:HumanoidEnv',
    max_episode_steps=1000,
)


register(
    id='MOAnt-v3',
    entry_point='mo_envs.mo_ant_v3:AntEnv',
    max_episode_steps=1000,
)

register(
    id='MOwalker2d-v3',
    entry_point='mo_envs.mo_walker2d_v3:Walker2dEnv',
    max_episode_steps=1000,
)

register(
    id='MOHopper-v3',
    entry_point='mo_envs.mo_hopper_v3:HopperEnv',
    max_episode_steps=1000,
)


register(
    id='MAant-v0',
    entry_point='mo_envs.MAant:AntEnv',
    max_episode_steps=1000,
)


register(
    id='MOfetch-v0',
    entry_point='mo_envs.mo_fetch:mo_FetchPickAndPlaceEnv',
    max_episode_steps=50,
)

register(
    id='MOpush-v0',
    entry_point='mo_envs.mo_push:mo_FetchPushEnv',
    max_episode_steps=50,
)

register(
    id='moAlleDoor-v0',
    entry_point='mo_envs.mo_alle_door:moAlleDoorEnv',
    max_episode_steps=50,
)

register(
    id='moAlleSlider-v0',
    entry_point='mo_envs.mo_alle_slider:moAlleSliderEnv',
    max_episode_steps=50,
)

register(
    id='moAllePush-v0',
    entry_point='mo_envs.mo_alle_push:moAllePushEnv',
    max_episode_steps=50,
)

register(
    id='moAlleReach-v0',
    entry_point='mo_envs.mo_alle_reach:moAlleReachEnv',
    max_episode_steps=50,
)

register(
    id='moAlleDrawer-v0',
    entry_point='mo_envs.mo_alle_drawer:moAlleDrawerEnv',
    max_episode_steps=50,
)

register(
    id='moAlleBasket-v0',
    entry_point='mo_envs.mo_alle_basket:moAlleBasketEnv',
    max_episode_steps=50,
)

register(
    id='moAlleDrone-v0',
    entry_point='mo_envs.mo_alle_drone:moAlleDroneEnv',
    max_episode_steps=50,
)

register(
    id='moAlleGolf-v0',
    entry_point='mo_envs.mo_alle_golf:moAlleGolfEnv',
    max_episode_steps=50,
)

register(
    id='moAlleBatting-v0',
    entry_point='mo_envs.mo_alle_batting:moAlleBattingEnv',
    max_episode_steps=50,
)

register(
    id='moRG6tray-v0',
    entry_point='mo_envs.moRG6tray:moRG6trayEnv',
    max_episode_steps=50,
)

register(
    id='moRG6Golf-v0',
    entry_point='mo_envs.moRG6Golf:moRG6GolfEnv',
    max_episode_steps=50,
)

register(
    id='moRG6classify-v0',
    entry_point='mo_envs.moRG6classify:moRG6classifyEnv',
    max_episode_steps=50,
)

register(
    id='Schpaint-v0',
    entry_point='mo_envs.Schpaint:SchpaintEnv',
    max_episode_steps=50,
)

register(
    id='Schmobox-v0',
    entry_point='mo_envs.Schmobox:SchmoboxEnv',
    max_episode_steps=50,
)

register(
    id='Schmopillar-v0',
    entry_point='mo_envs.Schmopillar:SchmopillarEnv',
    max_episode_steps=50,
)

register(
    id='cqlSchmopillar-v0',
    entry_point='mo_envs.cqlSchmopillar:SchmopillarEnv',
    max_episode_steps=50,
)

register(
    id='cqlMOpush-v0',
    entry_point='mo_envs.cql_mo_push:mo_FetchPushEnv',
    max_episode_steps=50,
)