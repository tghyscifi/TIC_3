print_memory = False    #默認不打印
print_mem_map = {
    # chip_x, chip_y, core_x, core_y
    (0, 0, 0, 0) : [
        {
            "step": 0,
            "phase": 3,
            "start_addr": 0,
            "data_length": 1024
        },
        {
            "step": 0,
            "phase": 0,
            "start_addr": 33794,
            "data_length": 32
        }
    ],
    (0, 0, 1, 0) : [
        {
            "step": 0,
            "phase": 0,
            "start_addr": 33792,
            "data_length": 256
        },
    ],
}