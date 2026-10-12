import numpy as np

Cin = 28
Cout = 28
cache_line = 2
insert_line = 1
Kw = 3
W_full = 1920
H_full = 1080

def mem_w(cin, cout, k):
    return np.ceil(cout/32) * 32 * cin * k * k

def mem_fin(cin, wfin, hfin):
    return np.ceil(cin/16) * 16 * wfin * hfin

def mem_fout(cout, wfout, hfout):
    return np.ceil(cout / 16) * 16 * wfout * hfout

def mem_bias(cout):
    return np.ceil(cout/32) * 32 * 4

def mem_mid(cout, wfout, cache):
    return wfout * cache * np.ceil(cout/32) * 32 * 4

if __name__ == "__main__":
    least_step = W_full * H_full
    w_out_least = 0
    h_out_least = 0
    cache_least = 0

    # 参数是固定的
    Mem_w = mem_w(Cin, Cout, Kw)
    Mem_b = mem_bias(Cout)

    # 遍历所有的长宽选择
    for i in range(17, W_full + 1):
        w_in = i
        w_out = i
        if (Mem_w * 2 + Mem_b * 2 + mem_fin(Cin, w_in, 1) * 2 + mem_fout(Cout, w_out, 1) + mem_mid(Cout, w_out, 2)) / 1024 > 128\
                or (Mem_b + Mem_w + mem_fin(Cin, w_in, 1) + mem_fout(Cout, w_out, 1)) / 1024 > 64:
            print(w_out)
            print(Mem_w/1024)
            print(Mem_b/1024)
            print(mem_fin(Cin, w_in, 1)/1024)
            print(mem_fout(Cout, w_out, 1)/1024)
            print(mem_mid(Cout, w_out, 2)/1024)
            break
        for j in range(1, H_full + 1):
            h_in = j + 2
            h_out = j
            if (Mem_w * 2 + Mem_b * 2 + mem_fin(Cin, w_in, h_in) * 2 + mem_fout(Cout, w_out, h_out) + mem_mid(Cout, w_out, 2)) / 1024 > 128\
                    or (Mem_b + Mem_w + mem_fin(Cin, w_in, h_in) + mem_fout(Cout, w_out, h_out) - mem_fin(Cin, w_in, 1)) / 1024 > 64:
                break
            for kk in range(2, h_out):
                if (Mem_w * 2 + Mem_b * 2 + mem_fin(Cin, w_in, h_in) * 2 + mem_fout(Cout, w_out, h_out) + mem_mid(Cout, w_out, kk)) / 1024 > 128:
                    break
                else:
                    step_num = np.ceil(W_full/(w_out-16)) * np.ceil(H_full/(h_out*40 - 16))
                    if step_num <= least_step:
                        least_step = step_num.copy()
                        w_out_least = w_out
                        h_out_least = h_out
                        cache_least = kk
                        print("step:", step_num)
                        print(i, j, kk)
                    # else:
                    #     print("step:", step_num)
                    #     print(i, j, kk)

    print("least step:", least_step)
    print(w_out_least, h_out_least, cache_least)

