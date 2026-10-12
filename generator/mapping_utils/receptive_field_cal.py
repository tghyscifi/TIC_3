def receptive_field_cal_1d(layers: list, final_neurons: list):
    """
        反向计算感受野的大小
        parameters:
            layers          : describe all layers
                            [
                                {
                                    'kernel': 8,
                                    'pad': (1, 0),
                                    'stride': 7,
                                    'dilation': 1
                                },
                                ...
                            ]
            final_neurons   : [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36)]
    """
    receptive_field = [final_neurons]
    for idx in range(len(layers)):
        layer = layers[len(layers) - 1 - idx]
        kernel, pad, stride, dilation = layer['kernel'], layer['pad'], layer['stride'], layer['dilation']
        last_field, new_field = receptive_field[-1], []
        for interval in last_field:
            new_field.append(
                (interval[0] * stride - pad[0], (interval[1] - 1) * stride + dilation * (kernel - 1) + 1 - pad[0]))
        receptive_field.append(new_field)
    return receptive_field


if __name__ == '__main__':
    layers = [
        {
            'kernel': 3,
            'pad': (0, 0),
            'stride': 1,
            'dilation': 1
        },
        {
            'kernel': 8,
            'pad': (0, 0),
            'stride': 7,
            'dilation': 1
        }
    ]

    y = receptive_field_cal_1d(layers=layers, final_neurons=[(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36)])

    print(y)
