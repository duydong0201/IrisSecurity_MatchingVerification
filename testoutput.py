import numpy as np

# load lại file mã
code = np.load("codes/person1_L_S5001L00.npy")

print("Chiều dài IrisCode:", len(code))
print("50 bit đầu:", code[:500])
