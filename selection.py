from shutil import copy, move
from pathlib import Path

main = Path("./results/test_main")
aux = Path("./results/test_aux")

aux_in = """1111255_67e9909adc_b.jpg
12823872185_c45a4ab1f9_b.jpg
1450001932_58913f9f3a_b.jpg
15570431986_26c5a73ca1_b.jpg
2164476221_beb23e7d93_b.jpg
2164699338_681feb5369_b.jpg
22337455078_1ab2d18a70_b.jpg
2288962505_a265fd7637_b.jpg
2303097732_5af449a068_b.jpg
2457385_2a7d799ebd_b.jpg
2471046601_2003bc6d99_b.jpg
26074978288_85caf9c745_b.jpg
2700797410_eb7c87f70e_b.jpg
29909090356_1b1ca6f6b2_b.jpg
300308951_e822078746_b.jpg
3312558905_ebeb880684_b.jpg
3534119373_d4cb4d5bd2_b.jpg
40751423_2f7601a414_b.jpg
424321058_071fd4c636_b.jpg
43043693910_3a24bb10e9_b.jpg
43297801115_b971bf74c2_b.jpg
4480781671_c9dff66a53_b.jpg
47055654881_c3eef02631_b.jpg
49968542737_1aa7dd5c41_b.jpg
5130188290_bcbfb1ed3f_b.jpg
5289738793_cecbe7b5c6_b.jpg
5339165647_a9b078f877_b.jpg
5349864310_0ecec86db7_b.jpg
5729066518_19b8c453d9_b.jpg
6158400930_388993b941_b.jpg
6802631028_8811de865d_b.jpg
8622709078_a253454f94_b.jpg
8629709091_5e264920f0_b.jpg
8629714255_51610a5c38_b.jpg
9666833536_aeba022e2c_b.jpg"""

aux_in = aux_in.split("\n")

for item in aux_in:
    try:
        copy(aux / item, main / item)
    except Exception as e:
        print(e)

move(main, main.parent / "test_mixed")
