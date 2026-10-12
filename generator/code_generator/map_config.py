from typing import Dict


class GroupConfig(object):
    def __init__(self, group_config):
        self._group_config = group_config
        self._core_cofig = {}
        for key, value in self._group_config.items():
            if isinstance(key, tuple):
                self._core_cofig[key] = value

    @property
    def clock(self):
        return self._group_config.get('clock')

    @property
    def mode(self):
        return self._group_config.get('mode')

    @property
    def mode(self):
        return self._group_config.get('mode')

    @property
    def clock0_in_step(self):
        return self._group_config.get('clock0_in_step')

    @property
    def clock1_in_step(self):
        return self._group_config.get('clock1_in_step')

    @property
    def core_list(self):
        return self._core_cofig.keys()

    def get_instant_pi_list(self, core_id):
        return self._core_cofig[core_id].get('instant_prims', [])

    def get_prim_list(self, core_id, prim_name, index):
        if 'prims' in self._core_cofig.get(core_id, []):
            prims = self._core_cofig[core_id]['prims']
            one_list = []
            for pi_group in prims:
                if isinstance(pi_group, dict):
                    one_pi = pi_group.get(prim_name, None)
                    if isinstance(one_pi, list):
                        one_list.append(one_pi[0])
                    else:
                        one_list.append(one_pi)
                else:
                    if isinstance(pi_group[index], list):
                        one_list.append(pi_group[index][0])
                    else:
                        one_list.append(pi_group[index])
            return one_list
        return self._core_cofig.get(core_id, []).get(prim_name, [])

    def axon_list(self, core_id):
        return self.get_prim_list(core_id, 'axon', 0)

    def soma1_list(self, core_id):
        return self.get_prim_list(core_id, 'soma1', 1)

    def router_list(self, core_id):
        return self.get_prim_list(core_id, 'router', 2)

    def soma2_list(self, core_id):
        return self.get_prim_list(core_id, 'soma2', 3)

    def get_registers(self, core_id):
        return self._core_cofig.get(core_id, {}).get("registers", {})


class MapConfig(object):
    def __init__(self, map_config):
        self._map_config = {}     # type: Dict[str, GroupConfig]
        for step_id, step_config in map_config.items():
            if isinstance(step_id, int):
                step_id = ((0, 0), step_id)
            if isinstance(step_id, str):
                if step_id == 'step_clock':
                    self._map_config['step_clock'] = {}
                    for chip_id_trigger, clock in step_config.items():
                        self._map_config['step_clock'][chip_id_trigger] = clock
                    continue
                self._map_config[step_id] = step_config
                continue
            for group_id, group_config in step_config.items():
                if isinstance(group_id, str):
                    if str(step_id) in self._map_config:
                        self._map_config[str(step_id)][group_id] = group_config
                    else:
                        self._map_config[str(step_id)] = {}
                        self._map_config[str(step_id)][group_id] = group_config
                    continue
                if not isinstance(group_id, str):
                    id = (step_id, group_id)
                    self._map_config[id] = GroupConfig(group_config)

    def __iter__(self):
        for id, config in self._map_config.items():
            if isinstance(id, str):
                continue
            step_id, group_id = id
            yield step_id, group_id, config

    @property
    def sim_clock(self):
        return self._map_config.get('sim_clock', None)

    def get_cycles_number(self, step_id):
        return self._map_config.get(str(step_id), {}).get("step_exe_number", 1)

    # def get_clock0_in_step(self, step_id):
    #     return self._map_config.get(str(step_id), {}).get("clock0_in_step", None)

    # def get_clock1_in_step(self, step_id):
    #     return self._map_config.get(str(step_id), {}).get("clock1_in_step", None)

    def get_trigger_clock(self, chip_id_trigger):
        return self._map_config.get('step_clock', {}).get(chip_id_trigger, (None, None))
