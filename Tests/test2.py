MyModel((_body): Sequential((0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))(1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)(2): Dropout(p=0.1, inplace=False)(3): ReLU(inplace=True)(4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)(5): Conv2d(16, 20, kernel_size=(2, 2), stride=(1, 1))(6): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)(7): Dropout(p=0.1, inplace=False)(8): ReLU(inplace=True)(9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)(10): Conv2d(20, 50, kernel_size=(2, 2), stride=(1, 1))(11): BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)(12): Dropout(p=0.1, inplace=False)(13): ReLU(inplace=True)(14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)(15): Conv2d(50, 200, kernel_size=(2, 2), stride=(1, 1))(16): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)(17): Dropout(p=0.1, inplace=False)(18): ReLU(inplace=True)(19): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))(_head): Sequential((0): Linear(in_features=200, out_features=100, bias=True)(1): Dropout(p=0.1, inplace=False)(2): ReLU(inplace=True)(3): Linear(in_features=100, out_features=60, bias=True)(4): Dropout(p=0.1, inplace=False)(5): ReLU(inplace=True)(6): Linear(in_features=60, out_features=10, bias=True)))