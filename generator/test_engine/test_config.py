
class TestConfig(object):
    def __init__(self, config_dict):
        self._config_dict = config_dict

    @property
    def tb_name(self):
        return self._config_dict['tb_name']

    @property
    def test_mode(self):
        return self._config_dict['test_mode']

    @property
    def test_group_phase(self):
        return self._config_dict['test_group_phase']

    def debug_file_switch(self):
        return self._config_dict.get('debug_file_switch', None)


class HardwareDebugFileSwitch(object):
    def __init__(self):
        self.MCHIP_CASE = 0
        self.DUMP_A_AD_INA = 1
        self.DUMP_A_DI_INA = 1
        self.DUMP_A_DO_INA = 1
        self.DUMP_A_AD_INB = 1
        self.DUMP_A_DI_INB = 1
        self.DUMP_A_DO_INB = 1
        self.DUMP_A_AD_BIAS = 1
        self.DUMP_A_DI_BIAS = 1
        self.DUMP_A_DO_BIAS = 1
        self.DUMP_A_AD_VOU = 1
        self.DUMP_A_DO_VOU = 1
        self.DUMP_MAC = 1
        self.DUMP_D_SOMA = 1
        self.DUMP_A_RHEAD = 1
        self.DUMP_D_RHEAD = 1
        self.DUMP_A_DOUT = 1
        self.DUMP_D_DOUT = 1
        self.DUMP_D_SEND = 1
        self.DUMP_A_DIN = 1
        self.DUMP_D_DIN = 1
        self.DUMP_DBGMSG_T = 1
        self.DUMP_DBGMSG_C = 1
        self.DUMP_DBGMSG_A = 1
        self.DUMP_DBGMSG_S = 1
        self.DUMP_DBGMSG_R = 1
        self.DUMP_STEP_NUM = 2
        self.DUMP_PHASE_NUM = 17
        self.BURST_CONFIG = 0
        self.BURST_READ = 0

    @property
    def open_burst(self):
        self.BURST_CONFIG = 1
        self.BURST_READ = 1
        return self

    @property
    def close_burst(self):
        self.BURST_CONFIG = 0
        self.BURST_READ = 0
        return self

    @property
    def open_debug_message(self):
        self.DUMP_DBGMSG_T = 1
        self.DUMP_DBGMSG_C = 1
        self.DUMP_DBGMSG_A = 1
        self.DUMP_DBGMSG_S = 1
        self.DUMP_DBGMSG_R = 1
        return self

    @property
    def dict(self):
        return self.__dict__

    @property
    def singla_chip(self):
        self.MCHIP_CASE = 0
        return self

    @property
    def multi_chip(self):
        self.MCHIP_CASE = 1
        return self

    @property
    def close_debug_message(self):
        self.DUMP_DBGMSG_T = 0
        self.DUMP_DBGMSG_C = 0
        self.DUMP_DBGMSG_A = 0
        self.DUMP_DBGMSG_S = 0
        self.DUMP_DBGMSG_R = 0
        return self

    @property
    def close_all(self):
        self.DUMP_A_AD_INA = 0
        self.DUMP_A_DI_INA = 0
        self.DUMP_A_DO_INA = 0
        self.DUMP_A_AD_INB = 0
        self.DUMP_A_DI_INB = 0
        self.DUMP_A_DO_INB = 0
        self.DUMP_A_AD_BIAS = 0
        self.DUMP_A_DI_BIAS = 0
        self.DUMP_A_DO_BIAS = 0
        self.DUMP_A_AD_VOU = 0
        self.DUMP_A_DO_VOU = 0
        self.DUMP_MAC = 0
        self.DUMP_D_SOMA = 0
        self.DUMP_A_RHEAD = 0
        self.DUMP_D_RHEAD = 0
        self.DUMP_A_DOUT = 0
        self.DUMP_D_DOUT = 0
        self.DUMP_D_SEND = 0
        self.DUMP_A_DIN = 0
        self.DUMP_D_DIN = 0
        self.DUMP_DBGMSG_T = 0
        self.DUMP_DBGMSG_C = 0
        self.DUMP_DBGMSG_A = 0
        self.DUMP_DBGMSG_S = 0
        self.DUMP_DBGMSG_R = 0
        return self
