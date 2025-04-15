import numpy as np
from sko.PSO import PSO
import matplotlib.pyplot as plt
from math import sin, cos

# Set printing options: suppress scientific notation and set 8 decimal places
np.set_printoptions(suppress=True, precision=4)

# Basic data
n_stations = 5  # Number of hydropower stations
n_hours = 24  # Hours per station

# Hydropower inflow data (5 stations × 24 hours)
water_data = np.array([
    [1.862681785, 1.858936215, 1.85479762, 1.850526163, 1.846380257, 1.842474011, 1.838806538, 1.835707889, 1.832940994,
     1.829780642, 1.826329883, 1.822598024, 1.819098103, 1.815427513, 1.811311224, 1.807367106, 1.803212314, 1.79849099,
     1.793174498, 1.787395324, 1.78159679, 1.77585077, 1.770860612, 1.7656539],
    [
        3.787201488, 3.774432138, 3.761621606, 3.748727272, 3.736081741, 3.723755542, 3.711595778, 3.699838335,
        3.689353662, 3.678777167, 3.667742152, 3.656073065, 3.646189856, 3.635636871, 3.625150513, 3.615054505,
        3.604936673, 3.593377397, 3.582072381, 3.570864701, 3.55874928, 3.545399196, 3.533125795, 3.520440448],
    [
        1.158050311, 1.158005405, 1.181767539, 1.181145228, 1.180726205, 1.18012744, 1.179252348, 1.178279531,
        1.177570876, 1.177746697, 1.177880069, 1.177393436, 1.176681437, 1.175742537, 1.175101724, 1.173895639,
        1.172506919, 1.171843657, 1.171046599, 1.171135317, 1.169789055, 1.168779033, 1.05375844, 1.054270385],
    [
        0.34989575, 0.348089843, 0.346197813, 0.344235608, 0.342370929, 0.340694086, 0.3392466, 0.337776199,
        0.336382983, 0.335091165, 0.333918185, 0.332823967, 0.332060594, 0.331401025, 0.330710725, 0.329675667,
        0.328355053, 0.326871502, 0.325378536, 0.323940801, 0.322394997, 0.320817531, 0.319345125, 0.317961556],
    [
        1.037253556, 1.033686504, 1.049944085, 1.045956121, 1.042128374, 1.038493405, 1.035388978, 1.032382541,
        1.029873763, 1.027195959, 1.024877853, 1.02226799, 1.020049944, 1.017661208, 1.015356729, 1.012606995,
        1.0095677, 1.006401466, 1.003337111, 1.000808939, 0.997929168, 0.994905214, 0.973294846, 0.971309598]
])

# Load data
load_set = np.array(
    [73.30904235, 73.30904235, 73.30904235, 73.30904235, 73.30904235, 73.30904235, 73.30904235, 78.77193723,
     92.60998531, 96.70431357, 110.7230375, 110.7230375, 99.39242037, 99.39242037, 99.39242037, 99.39242037,
     95.48471433, 95.48471433, 95.48471433, 95.48471433, 95.48471433, 108.283839, 102.1875485, 93.65205957]
)

# Upstream water level curves (5 stations × multiple data points)
Z_up_data = np.array(
    [[480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
      547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569,
      570],
     [271, 271.1, 271.2, 271.3, 271.4, 271.5, 271.6, 271.7, 271.8, 271.9, 272, 272.1, 272.2, 272.3, 272.4, 272.5, 272.6,
      272.7, 272.8, 272.9, 273, 273.1, 273.2, 273.3, 273.4, 273.5, 273.6, 273.7, 273.8, 273.9, 274, 274.1, 274.2, 274.3,
      274.4, 274.5, 274.6, 274.7, 274.8, 274.9, 275, 275.1, 275.2, 275.3, 275.4, 275.5, 275.6, 275.7, 275.8, 275.9, 276,
      276.1, 276.2, 276.3, 276.4, 276.5, 276.6, 276.7, 276.8, 276.9, 277, 277.1, 277.2, 277.3, 277.4, 277.5, 277.6,
      277.7, 277.8, 277.9, 278, 278.1, 278.2, 278.3, 278.4, 278.5, 278.6, 278.7, 278.8, 278.9, 279, 279.1, 279.2, 279.3,
      279.4, 279.5, 279.6, 279.7, 279.8, 279.9, 280, 280.1, 280.2, 280.3, 280.4, 280.5, 280.6, 280.7, 280.8, 280.9, 281,
      281.1, 281.2, 281.3, 281.4, 281.5, 281.6, 281.7, 281.8, 281.9, 282, 282.1, 282.2, 282.3, 282.4, 282.5, 282.6,
      282.7, 282.8, 282.9, 283, 283.1, 283.2, 283.3, 283.4, 283.5, 283.6, 283.7, 283.8, 283.9, 284, 284.1, 284.2, 284.3,
      284.4, 284.5, 284.6, 284.7, 284.8, 284.9, 285, 285.1, 285.2, 285.3, 285.4, 285.5, 285.6, 285.7, 285.8, 285.9, 286,
      286.1, 286.2, 286.3, 286.4, 286.5, 286.6, 286.7, 286.8, 286.9, 287, 287.1, 287.2, 287.3, 287.4, 287.5, 287.6,
      287.7, 287.8, 287.9, 288, 288.1, 288.2, 288.3, 288.4, 288.5, 288.6, 288.7, 288.8, 288.9, 289, 289.1, 289.2, 289.3,
      289.4, 289.5, 289.6, 289.7, 289.8, 289.9, 290, 290.1, 290.2, 290.3, 290.4, 290.5, 290.6, 290.7, 290.8, 290.9, 291,
      291.1, 291.2, 291.3, 291.4, 291.5, 291.6, 291.7, 291.8, 291.9, 292, 292.1, 292.2, 292.3, 292.4, 292.5, 292.6,
      292.7, 292.8, 292.9, 293, 293.1, 293.2, 293.3, 293.4, 293.5, 293.6, 293.7, 293.8, 293.9, 294, 294.1, 294.2, 294.3,
      294.4, 294.5, 294.6, 294.7, 294.8, 294.9, 295, 295.1, 295.2, 295.3, 295.4, 295.5, 295.6, 295.7, 295.8, 295.9, 296,
      296.1, 296.2, 296.3, 296.4, 296.5, 296.6, 296.7, 296.8, 296.9, 297, 297.1, 297.2, 297.3, 297.4, 297.5, 297.6,
      297.7, 297.8, 297.9, 298, 298.1, 298.2, 298.3, 298.4, 298.5, 298.6, 298.7, 298.8, 298.9, 299, 299.1, 299.2, 299.3,
      299.4, 299.5, 299.6, 299.7, 299.8, 299.9, 300, 300.1, 300.2, 300.3, 300.4, 300.5, 300.6, 300.7, 300.8, 300.9, 301,
      301.1, 301.2, 301.3, 301.4, 301.5, 301.6, 301.7, 301.8, 301.9, 302, 302.1, 302.2, 302.3, 302.4, 302.5, 302.6,
      302.7, 302.8, 302.9, 303, 303.1, 303.2, 303.3, 303.4, 303.5, 303.6, 303.7, 303.8, 303.9, 304, 304.1, 304.2, 304.3,
      304.4, 304.5, 304.6, 304.7, 304.8, 304.9, 305, 305.1, 305.2, 305.3, 305.4, 305.5, 305.6, 305.7, 305.8, 305.9, 306,
      306.1, 306.2, 306.3, 306.4, 306.5, 306.6, 306.7, 306.8, 306.9, 307, 307.1, 307.2, 307.3, 307.4, 307.5, 307.6,
      307.7, 307.8, 307.9, 308, 308.1, 308.2, 308.3, 308.4, 308.5, 308.6, 308.7, 308.8, 308.9, 309, 309.1, 309.2, 309.3,
      309.4, 309.5, 309.6, 309.7, 309.8, 309.9, 310, 310.1, 310.2, 310.3, 310.4, 310.5, 310.6, 310.7, 310.8, 310.9, 311,
      311.1, 311.2, 311.3, 311.4, 311.5, 311.6, 311.7, 311.8, 311.9, 312, 312.1, 312.2, 312.3, 312.4, 312.5, 312.6,
      312.7, 312.8, 312.9, 313, 313.1, 313.2, 313.3, 313.4, 313.5, 313.6, 313.7, 313.8, 313.9, 314, 314.1, 314.2, 314.3,
      314.4, 314.5, 314.6, 314.7, 314.8, 314.9, 315, 315.1, 315.2, 315.3, 315.4, 315.5, 315.6, 315.7, 315.8, 315.9, 316,
      316.1, 316.2, 316.3, 316.4, 316.5, 316.6, 316.7, 316.8, 316.9, 317, 317.1, 317.2, 317.3, 317.4, 317.5, 317.6,
      317.7, 317.8, 317.9, 318, 318.1, 318.2, 318.3, 318.4, 318.5, 318.6, 318.7, 318.8, 318.9, 319, 319.1, 319.2, 319.3,
      319.4, 319.5, 319.6, 319.7, 319.8, 319.9, 320, 320.1, 320.2, 320.3, 320.4, 320.5, 320.6, 320.7, 320.8, 320.9, 321,
      321.1, 321.2, 321.3, 321.4, 321.5, 321.6, 321.7, 321.8, 321.9, 322, 322.1, 322.2, 322.3, 322.4, 322.5, 322.6,
      322.7, 322.8, 322.9, 323, 323.1, 323.2, 323.3, 323.4, 323.5, 323.6, 323.7, 323.8, 323.9, 324, 324.1, 324.2, 324.3,
      324.4, 324.5, 324.6, 324.7, 324.8, 324.9, 325, 325.1, 325.2, 325.3, 325.4, 325.5, 325.6, 325.7, 325.8, 325.9, 326,
      326.1, 326.2, 326.3, 326.4, 326.5, 326.6, 326.7, 326.8, 326.9, 327, 327.1, 327.2, 327.3, 327.4, 327.5, 327.6,
      327.7, 327.8, 327.9, 328, 328.1, 328.2, 328.3, 328.4, 328.5, 328.6, 328.7, 328.8, 328.9, 329, 329.1, 329.2, 329.3,
      329.4, 329.5, 329.6, 329.7, 329.8, 329.9, 330, 330.1, 330.2, 330.3, 330.4, 330.5, 330.6, 330.7, 330.8, 330.9, 331,
      331.1, 331.2, 331.3, 331.4, 331.5, 331.6, 331.7, 331.8, 331.9, 332, 332.1, 332.2, 332.3, 332.4, 332.5, 332.6,
      332.7, 332.8, 332.9, 333, 333.1, 333.2, 333.3, 333.4, 333.5, 333.6, 333.7, 333.8, 333.9, 333.99],
     [180.5, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 201, 202, 203, 204, 205, 206],
     [569.7, 570, 571, 572, 573, 574, 575, 576, 577],
     [597, 598, 600, 602, 604, 606, 608, 610, 612, 614, 616, 618, 620, 622, 624, 626, 628, 630, 632, 634, 636, 638, 640,
      642, 644, 646]], dtype=object)  # Truncated for brevity

# Reservoir capacity curves (5 stations × multiple data points, unit: m³)
V_hydro_data = np.array(
    [[0, 21300, 85100, 191400, 340300, 556700, 839400, 1217000, 1678800, 2230800, 2885300, 3619000, 3780000, 3950000,
      4120000, 4290000, 4484200, 4666800, 4855000, 5048800, 5248300, 5453400, 5660500, 5875100, 6097100, 6326500,
      6563400, 6812600, 7066800, 7325900, 7590000, 7859100, 8132400, 8411000, 8695000, 8984200, 9278800, 9576900,
      9881300, 10191800, 10508500, 10831500, 11161400, 11497100, 11838700, 12186100, 12539300],
     [12500000, 12580000, 12650000, 12730000, 12810000, 12890000, 12960000, 13040000, 13120000, 13190000, 13270000,
      13350000, 13440000, 13520000, 13610000, 13690000, 13770000, 13860000, 13940000, 14030000, 14110000, 14200000,
      14290000, 14380000, 14470000, 14570000, 14660000, 14750000, 14840000, 14930000, 15020000, 15120000, 15220000,
      15310000, 15410000, 15510000, 15610000, 15710000, 15800000, 15900000, 16000000, 16100000, 16200000, 16300000,
      16400000, 16500000, 16600000, 16700000, 16800000, 16900000, 17000000, 17100000, 17210000, 17310000, 17410000,
      17520000, 17620000, 17720000, 17820000, 17930000, 18030000, 18140000, 18240000, 18350000, 18450000, 18560000,
      18670000, 18770000, 18880000, 18980000, 19090000, 19200000, 19310000, 19420000, 19530000, 19640000, 19740000,
      19850000, 19960000, 20070000, 20180000, 20290000, 20400000, 20520000, 20630000, 20740000, 20850000, 20960000,
      21080000, 21190000, 21300000, 21420000, 21530000, 21650000, 21760000, 21880000, 21990000, 22110000, 22220000,
      22340000, 22450000, 22550000, 22650000, 22750000, 22850000, 22950000, 23050000, 23150000, 23250000, 23350000,
      23630000, 23750000, 23870000, 24000000, 24120000, 24240000, 24360000, 24480000, 24610000, 24730000, 24850000,
      24980000, 25100000, 25230000, 25350000, 25480000, 25610000, 25730000, 25860000, 25980000, 26110000, 26240000,
      26370000, 26500000, 26630000, 26760000, 26880000, 27010000, 27140000, 27270000, 27400000, 27530000, 27660000,
      27800000, 27930000, 28060000, 28190000, 28320000, 28460000, 28590000, 28720000, 28850000, 28990000, 29120000,
      29260000, 29390000, 29520000, 29660000, 29790000, 29930000, 30060000, 30200000, 30330000, 30470000, 30600000,
      30740000, 30880000, 31010000, 31150000, 31280000, 31420000, 31560000, 31700000, 31830000, 31970000, 32110000,
      32250000, 32390000, 32520000, 32660000, 32800000, 32940000, 33080000, 33220000, 33360000, 33500000, 33640000,
      33780000, 33920000, 34060000, 34200000, 34340000, 34490000, 34630000, 34780000, 34920000, 35060000, 35210000,
      35350000, 35500000, 35710000, 35870000, 36020000, 36180000, 36330000, 36490000, 36640000, 36800000, 36950000,
      37110000, 37260000, 37420000, 37580000, 37740000, 37900000, 38060000, 38220000, 38380000, 38540000, 38700000,
      38860000, 39020000, 39190000, 39350000, 39520000, 39680000, 39840000, 40010000, 40170000, 40340000, 40500000,
      40670000, 40840000, 41010000, 41180000, 41350000, 41510000, 41680000, 41850000, 42020000, 42190000, 42360000,
      42540000, 42710000, 42890000, 43060000, 43230000, 43410000, 43580000, 43760000, 43930000, 44110000, 44290000,
      44460000, 44640000, 44820000, 45000000, 45180000, 45350000, 45530000, 45710000, 45890000, 46070000, 46240000,
      46420000, 46600000, 46780000, 46960000, 47130000, 47310000, 47490000, 47680000, 47870000, 48050000, 48240000,
      48430000, 48620000, 48810000, 48990000, 49180000, 49370000, 49560000, 49750000, 49950000, 50140000, 50330000,
      50520000, 50720000, 50910000, 51100000, 51300000, 51490000, 51690000, 51880000, 52080000, 52280000, 52470000,
      52670000, 52870000, 53070000, 53260000, 53460000, 53660000, 53860000, 54060000, 54260000, 54460000, 54660000,
      54860000, 55060000, 55270000, 55470000, 55670000, 55870000, 56080000, 56280000, 56490000, 56690000, 56890000,
      57100000, 57310000, 57510000, 57720000, 57920000, 58130000, 58340000, 58550000, 58750000, 58960000, 59170000,
      59380000, 59590000, 59800000, 60010000, 60220000, 60430000, 60640000, 60850000, 61060000, 61270000, 61480000,
      61700000, 61910000, 62120000, 62340000, 62550000, 62760000, 62980000, 63190000, 63410000, 63620000, 63840000,
      64050000, 64270000, 64490000, 64700000, 64920000, 65140000, 65350000, 65570000, 65790000, 66010000, 66230000,
      66450000, 66670000, 66890000, 67110000, 67330000, 67550000, 67770000, 68000000, 68220000, 68440000, 68660000,
      68880000, 69110000, 69330000, 69550000, 69780000, 70000000, 70230000, 70450000, 70680000, 70910000, 71130000,
      71360000, 71590000, 71810000, 72040000, 72270000, 72500000, 72730000, 72960000, 73190000, 73420000, 73650000,
      73880000, 74110000, 74340000, 74570000, 74810000, 75040000, 75270000, 75510000, 75740000, 75970000, 76210000,
      76440000, 76680000, 76920000, 77160000, 77390000, 77630000, 77870000, 78110000, 78340000, 78580000, 78820000,
      79060000, 79310000, 79550000, 79790000, 80030000, 80270000, 80520000, 80760000, 81000000, 81250000, 81500000,
      81740000, 81990000, 82230000, 82480000, 82730000, 82980000, 83230000, 83480000, 83730000, 83980000, 84230000,
      84480000, 84740000, 84990000, 85240000, 85500000, 85750000, 86010000, 86260000, 86520000, 86780000, 87040000,
      87300000, 87560000, 87820000, 88080000, 88340000, 88600000, 88870000, 89130000, 89400000, 89660000, 89930000,
      90190000, 90460000, 90730000, 91000000, 91270000, 91540000, 91810000, 92080000, 92360000, 92630000, 92900000,
      93180000, 93460000, 93730000, 94010000, 94290000, 94570000, 94850000, 95130000, 95420000, 95700000, 95980000,
      96270000, 96550000, 96840000, 97130000, 97420000, 97710000, 98000000, 98290000, 98580000, 98880000, 99170000,
      99470000, 99770000, 100060000, 100360000, 100660000, 100960000, 101270000, 101570000, 101870000, 102180000,
      102490000, 102790000, 103100000, 103410000, 103720000, 104040000, 104350000, 104660000, 104980000, 105300000,
      105610000, 105930000, 106250000, 106580000, 106900000, 107220000, 107490000, 107810000, 108120000, 108440000,
      108750000, 109070000, 109390000, 109710000, 110030000, 110360000, 110680000, 111010000, 111330000, 111660000,
      111980000, 112300000, 112630000, 112950000, 113280000, 113600000, 113960000, 114320000, 114680000, 115040000,
      115410000, 115770000, 116130000, 116490000, 116850000, 117210000, 117590000, 117980000, 118360000, 118740000,
      119130000, 119510000, 119890000, 120270000, 120660000, 121040000, 121440000, 121840000, 122250000, 122650000,
      123050000, 123450000, 123860000, 124260000, 124660000, 125060000, 125480000, 125900000, 126330000, 126740000,
      127170000, 127590000, 128010000, 128430000, 128850000, 129270000, 129700000, 130140000, 130570000, 131000000,
      131440000, 131870000, 132300000, 132730000, 133160000, 133600000, 134040000, 134470000, 134910000, 135340000,
      135780000, 136220000, 136650000, 137090000, 137520000, 137960000, 138410000, 138850000, 139300000, 139750000,
      140200000, 140640000, 141090000, 141540000, 141980000, 142430000, 142880000, 143340000, 143790000, 144250000,
      144700000, 145150000, 145610000, 146060000, 146520000, 146970000, 147420000, 147870000, 148320000, 148770000,
      149220000, 149670000, 150120000, 150570000, 151020000, 151430000],
     [0, 9200, 35500, 74800, 129800, 215000, 345000, 534000, 810000, 1256000, 1965000, 2429000, 2991000, 3643000,
      4425000, 5329000, 6407000],
     [181800, 192000, 236000, 280000, 334500, 389000, 456500, 524000, 603500],
     [0, 100, 1400, 4100, 10200, 20600, 33300, 48200, 67500, 92300, 121600, 155200, 196200, 247600, 310000, 381700,
      460400, 546900, 641000, 744100, 858200, 985000, 1127800, 1292400, 1483800, 1709000]],
    dtype=object)  # Truncated for brevity

# Tailwater level curves (5 stations × multiple data points)
Z_down_data = np.array([[456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466],
                        [230, 230.5, 231, 231.5, 232, 232.5, 233, 233.5, 234, 234.5, 235, 235.5],
                        [180.5, 181, 181.5, 182, 182.5, 183, 183.5, 184, 184.5, 185, 185.5, 186, 186.5, 187, 187.5, 188,
                         188.5, 189, 189.5,
                         190, 190.5, 191, 191.5, 192, 192.5, 193, 193.5],
                        [512, 512.5, 513, 513.5, 514, 514.5, 515, 515.5, 516],
                        [588.5, 590.5, 591, 591.5, 592, 592.5, 593, 593.5, 594, 594.5]],
                       dtype=object)  # Truncated for brevity

# Discharge flow curves (5 stations × multiple data points, unit: m³/s)
Q_out_data = np.array([[0, 12, 70.9, 167, 298, 464, 667, 907, 1189, 1515, 1885],
                       [0, 22.2, 55, 182, 390, 745, 1300, 2070, 3010, 4090, 5300, 6620],
                       [0.1, 7.6, 25, 62.5, 130, 224, 335, 465, 600, 740, 890, 1040, 1200, 1360, 1530, 1710, 1890, 2080,
                        2280, 2520, 2780,
                        3070, 3370, 3680, 3990, 4320, 4670],
                       [0, 9, 37, 51, 182, 425, 743, 1126, 1568],
                       [3.8, 17.15, 30, 77, 154, 280, 437, 629, 854, 1116]], dtype=object)  # Truncated for brevity

# Cascade hydropower parameters
I = 5  # Number of cascade stations

# Define time periods (assuming water_data is 2D array)
T = np.size(water_data, axis=1) if water_data is not None else 0  # Equivalent to MATLAB's size(water_data,2)
dt = 60  # Time interval (minutes)

# Water level parameters (unit: meters)
Z_up_max = np.array([570, 325, 203, 577, 633])  # Upper limits
Z_up_min = np.array([535, 298, 202.3, 569.7, 628])  # Lower limits
Z_up_begin = np.array([570, 325, 203, 577, 633])  # Initial control levels (wet season)
Z_up_end = np.array([570, 325, 203, 577, 633])  # Final control levels (wet season)

# Water head parameters (unit: meters)
H_max = np.array([114, 95, 22.7, 65, 44.5])  # Max water head
H_min = np.array([30, 30, 14, 40, 27])  # Min water head

# Reservoir capacity parameters (unit: m³)
V_max = np.array([12539300, 113600000, 3643000, 603500, 692550])  # Capacity limits

# Flow parameters (unit: m³/s)
Q_max = np.array([1060, 1500, 1000, 300, 200])  # Max discharge flow
Q_min = np.array([0, 0, 0, 0, 0])  # Min discharge flow
Q_amp = np.array([10.6, 75.694, 64.66, 5.2, 7.6]) * 0.2  # Discharge variation amplitude (normal season)

# Hydropower output parameters (unit: MW)
P_hydro_max = np.array([8.6, 49.8, 11.4, 1.26, 2.5])  # Max output
P_hydro_min = np.array([0, 0, 0, 0, 0])  # Min output

# Hydropower efficiency coefficients
A_hydro = np.array([8.3, 8.7, 8.7, 8, 8])

# Head loss parameters
H_loss_a = np.array([0, 0, 0, 0, 0])  # Head loss coefficients
H_loss_b = np.array([1, 1, 1, 0.5, 1])  # Head loss constants

# Power flow base data
n_bus = 12


# Power flow calculation data
def run_power_flow(buses, lines, max_iter=100, tolerance=1e-6, verbose=True):
    """Power flow calculation core function (Newton-Raphson method)"""
    # Deep copy inputs to preserve original data
    global P_inj, Q_inj
    buses = buses.copy(deep=True)
    lines = lines.copy(deep=True)

    # Initialize calculation data
    n_bus = len(buses[:, 0])
    V = buses[2].values.astype(float)
    theta = np.radians(buses[3].values.astype(float))
    Ybus = np.zeros((n_bus, n_bus), dtype=complex)

    # Build bus admittance matrix
    for _, line in lines.iterrows():
        i = line[0] - 1  # Convert to 0-based index
        j = line[1] - 1
        R, X, B = line[2], line[3], line[4]
        Z = R + 1j * X
        Y_series = 1 / Z
        Ybus[i, i] += Y_series + 1j * B / 2
        Ybus[j, j] += Y_series + 1j * B / 2
        Ybus[i, j] -= Y_series
        Ybus[j, i] -= Y_series

    # Newton-Raphson iteration
    for iter in range(max_iter):
        # Calculate power mismatches
        S_inj = V * (Ybus @ (V * np.exp(1j * theta))) * np.exp(-1j * theta)
        P_inj = S_inj.real
        Q_inj = S_inj.imag

        # Build mismatch vector
        mismatch = []
        for i in range(n_bus):
            bus_type = buses[1, i]
            if bus_type != 3:  # Non-slack buses
                P_spec = buses[6, i] - buses[4, i]
                mismatch.append(P_spec - P_inj[i])
                if bus_type == 1:  # PQ buses
                    Q_spec = buses[7, i] - buses[5, i]
                    mismatch.append(Q_spec - Q_inj[i])
        mismatch = np.array(mismatch)

        # Convergence check
        max_error = np.max(np.abs(mismatch))
        if verbose:
            print(f"Iter {iter}: Max mismatch = {max_error:.2e}")
        if max_error < tolerance:
            break

        # Build Jacobian matrix (polar coordinate version)
        J = np.zeros((len(mismatch), len(mismatch)))
        row = 0
        for i in range(n_bus):
            bus_type = buses[1, i]
            if bus_type != 3:  # Handle P equations
                for j in range(n_bus):
                    if buses[1, j] != 3:
                        # J11 = dP/dθ
                        J[row, j] = V[i] * V[j] * (Ybus[i, j].real * sin(theta[i] - theta[j])
                                                   - Ybus[i, j].imag * cos(theta[i] - theta[j]))
                if bus_type == 1:  # Handle Q equations
                    for j in range(n_bus):
                        # J21 = dQ/dθ
                        J[row + 1, j] = V[i] * V[j] * (-Ybus[i, j].real * cos(theta[i] - theta[j])
                                                       - Ybus[i, j].imag * sin(theta[i] - theta[j]))
                row += 1 + (bus_type == 1)

        # Solve correction equations
        delta = np.linalg.solve(J, -mismatch)

        # Update variables
        ptr = 0
        for i in range(n_bus):
            if buses[1, i] != 3:
                theta[i] += delta[ptr]
                ptr += 1
                if buses[1, i] == 1:
                    V[i] += delta[ptr]
                    ptr += 1

    # Generate results
    result = np.array([
        buses[0],  # Bus
        V.round(4),  # Vm
        np.degrees(theta).round(2),  # Va_deg
        P_inj.round(4),  # P_inj
        Q_inj.round(4)  # Q_inj
    ])
    return result


# Create an mpc dictionary
mpc = {
    "version": "2",
    "baseMVA": 100.0,
    "bus": np.array([
        [1, 3, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1, 1],
        [2, 2, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
        [3, 1, 90, 40, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
        [4, 2, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
        [5, 2, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
        [6, 2, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
        [7, 2, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
        [8, 2, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
        [9, 2, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
        [10, 2, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
        [11, 1, 45, 30, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
        [12, 2, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9]
    ]),
    "gen": np.array([
        [1, 0, 0, 0, 0, 1, 100, 1, 300, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 100, -50, 1, 100, 1, 49.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 13.4, -20, 1, 100, 1, 3.42, 3.42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 45.6, -20, 1, 100, 1, 11.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [6, 0, 0, 60.7, -20, 1, 100, 1, 28.6, 28.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [7, 0, 0, 28.5, -20, 1, 100, 1, 7.49, 7.49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, 0, 0, 30, -50, 1, 100, 1, 1.26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [9, 0, 0, 50, -20, 1, 100, 1, 2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 0, 0, 30.7, -30, 1, 100, 1, 8.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [12, 0, 0, 20.5, -10, 1, 100, 1, 5.42, 5.42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),
    "branch": np.array([
        [1, 4, 0.000294, 0.001031, 0, 125, 0, 0, 0, 0, 1, 0, 0],
        [1, 7, 0.001647, 0.075368, 0, 137.13, 0, 0, 1.0375, 0, 1, 0, 0],
        [2, 3, 0.05, 0.2, 0.02, 130, 130, 130, 0, 0, 1, -360, 360],
        [2, 4, 0.05, 0.12, 0.01, 70, 70, 70, 0, 0, 1, -360, 360],
        [4, 5, 0.000989, 0.05451, 0, 144.52, 0, 0, 1, 0, 1, 0, 0],
        [4, 10, 0.000294, 0.001031, 0, 40, 0, 0, 0, 0, 1, 0, 0],
        [5, 6, 0.000184, 0.013522, 0, 633.62, 0, 0, 1, 0, 1, 0, 0],
        [7, 8, 0.003616, 0.016546, 0.00308, 165.22, 0, 0, 0, 0, 1, 0, 0],
        [8, 9, 0.014094, 0.380702, 0, 45.5, 0, 0, 1, 0, 1, 0, 0],
        [10, 11, 0.004782, 0.152801, 0, 100.71, 0, 0, 1, 0, 1, 0, 0],
        [11, 12, 0.009547, 0.223636, 0, 54.86, 0, 0, 1, 0, 1, 0, 0]
    ])
}

# branch parameters
branch_f_bus = mpc["branch"][:, 0]
branch_t_bus = mpc["branch"][:, 1]
branch_r_bus = mpc["branch"][:, 2]

# power flow parameters
branch_num = len(mpc["branch"][:, 0])  # number of branches
bus_num = len(mpc["bus"][:, 0])  # number of nodes
v_i = np.ones((bus_num, T))  # node voltage magnitude
v_a = np.zeros((bus_num, T))  # node voltage phase angle

# adjustment of node voltage magnitude
mpc["bus"][:, 11] = 1.1  # Vmax
mpc["bus"][:, 12] = 0.9  # Vmin

# generator cost function
mpc["gencost"] = np.array([
    [2, 0, 0, 1, 0, 2, 0],
    [2, 0, 0, 1, 0, 2, 0],
    [2, 0, 0, 1, 0, 2, 0],
    [2, 0, 0, 1, 0, 2, 0],
    [2, 0, 0, 1, 0, 2, 0],
    [2, 0, 0, 1, 0, 2, 0],
    [2, 0, 0, 1, 0, 10, 0],
    [2, 0, 0, 1, 0, 20, 0]
])

# adjustment of reactance and resistance values
X = mpc["branch"][:, 3].copy()
R = mpc["branch"][:, 2].copy()
mpc["branch"][:, 3] = X / 2
mpc["branch"][:, 2] = R / 2


# interpolation calculation
# A_t is known, B_t is unknown
def A_B(A_set, B_set, A_t):
    n = len(A_set)
    if n <= 0:
        return 0
    elif n == 1:
        return B_set[0]

    i = 0
    while i < n - 1 and A_t > A_set[i]:
        i += 1

    x1 = A_set[i - 1]
    x2 = A_set[i]
    y1 = B_set[i - 1]
    y2 = B_set[i]

    B_t = y1 + (A_t - x1) / (x2 - x1) * (y2 - y1)
    return B_t


# calculation output function
def cal_N(station_num, Q1_fadian_set, Q1_out, Water1_data, Z1_begin, V1_max, H_loss, A_num, is_print=False):
    global Z_up_all_24
    V_Q_fadian_all = np.zeros(n_hours, dtype=float)  # Reservoir capacity sequence (generation flow only)
    V_all = np.zeros(n_hours, dtype=float)  # Reservoir capacity sequence (generation + spillage flow)
    Q_qi_all = np.zeros(n_hours, dtype=float)  # Spillage flow sequence
    Q_out_all = np.zeros(n_hours, dtype=float)  # Outflow sequence
    Z_wei_all = np.zeros(n_hours, dtype=float)  # Tailwater level sequence
    Z_up_all = np.zeros(n_hours, dtype=float)  # Upper reservoir level sequence
    H_all = np.zeros(n_hours, dtype=float)  # Head sequence
    N_all = np.zeros(n_hours, dtype=float)  # Power output sequence
    Q_in = np.zeros(n_hours, dtype=float)  # Inflow sequence

    # Calculate spillage flow
    for i in range(0, n_hours):
        if i == 0:
            V_Q_fadian_all[i] = A_B(Z_up_data[station_num], V_hydro_data[station_num], Z1_begin)
        else:
            # Only consider generation flow for spillage calculation
            V_Q_fadian_all_jia = V_Q_fadian_all[i - 1] + (
                    Q1_out[i - 1] + Water1_data[i - 1] - Q1_fadian_set[i - 1]) * 3600
            if V_Q_fadian_all_jia > V1_max:
                Q_qi_all[i - 1] = (V_Q_fadian_all_jia - V1_max) / 3600
                V_Q_fadian_all[i] = V1_max
            else:
                Q_qi_all[i - 1] = 0
                V_Q_fadian_all[i] = V_Q_fadian_all_jia

    # Integrated version
    for j in range(0, n_hours):
        if j == 0:
            # Calculate outflow
            Q_out_all[j] = Q1_fadian_set[j] + Q_qi_all[j]
            Z_wei_all[j] = A_B(Q_out_data[station_num], Z_down_data[station_num], Q_out_all[j])
            Q_in[j] = Q1_out[j] + Water1_data[j]

            # Calculate upper reservoir level and capacity sequence
            V_all[j] = A_B(Z_up_data[station_num], V_hydro_data[station_num], Z1_begin)
            Z_up_all[j] = Z1_begin
        else:
            # Calculate outflow
            Q_out_all[j] = Q1_fadian_set[j] + Q_qi_all[j]
            Z_wei_all[j] = A_B(Q_out_data[station_num], Z_down_data[station_num], Q_out_all[j])
            Q_in[j] = Q1_out[j] + Water1_data[j]

            # Calculate upper reservoir level and capacity sequence
            V_all[j] = V_all[j - 1] + (Q1_out[j - 1] + Water1_data[j - 1] - Q_out_all[j - 1]) * 3600
            Z_up_all[j] = A_B(V_hydro_data[station_num], Z_up_data[station_num], V_all[j])

    # Integrated version
    for m in range(0, n_hours):
        if m < n_hours - 1:
            # Calculate head sequence
            H_all[m] = round((Z_up_all[m] + Z_up_all[m + 1]) / 2 - Z_wei_all[m] - H_loss, 2)
            # Calculate power output sequence
            N_all[m] = round(A_num * Q1_fadian_set[m] * H_all[m] / 1000, 2)
        if m == n_hours - 1:
            # Calculate final reservoir capacity and level
            V_all_24 = V_all[n_hours - 1] + (
                    Q1_out[n_hours - 1] + Water1_data[n_hours - 1] - Q_out_all[n_hours - 1]) * 3600
            Z_up_all_24 = A_B(V_hydro_data[station_num], Z_up_data[station_num], V_all_24)
            H_all[m] = round((Z_up_all[m] + Z_up_all_24) / 2 - Z_wei_all[m] - H_loss, 2)
            # Calculate power output sequence
            N_all[m] = round(A_num * Q1_fadian_set[m] * H_all[m] / 1000, 2)

    if is_print:
        print(f'station_num: {station_num}')
        print(f'Q_out_all: {Q_out_all}')
        print(f'V_Q_fadian_all: {V_Q_fadian_all}')
        print(f'Q_qi_all: {Q_qi_all}')
        print(f'Z_up_all: {Z_up_all}')
        print(f'Z_wei_all: {Z_wei_all}')
        print(f'V_all: {V_all}')
        print(f'Q_in: {Q_in}')
        print(f'A_num: {A_num}')
        print(f'Q1_fadian_set: {Q1_fadian_set}')
        print(f'H_all: {H_all}')
        print('')
    else:
        pass

    return N_all, Q_out_all, Z_up_all, Z_up_all_24


def cal_N_beta(Q_fadian_set1, station_num=0, is_constant=True, is_print=True, is_N=True, is_Z=True,
               cal_N_is_print=False):
    Q_fadian_1 = Q_fadian_set1[0:24]
    Q_fadian_2 = Q_fadian_set1[24:48]
    Q_fadian_3 = Q_fadian_set1[48:72]
    Q_fadian_4 = Q_fadian_set1[72:96]
    Q_fadian_5 = Q_fadian_set1[96:120]

    Q_out_0 = np.zeros(n_hours, dtype=float)
    N_dianzhan_1, Q_out_1, Z_dianzhan_1, Z_dianzhan_1_24 = cal_N(station_num=0, Q1_fadian_set=Q_fadian_1,
                                                                 Q1_out=Q_out_0,
                                                                 Water1_data=water_data[0], Z1_begin=Z_up_begin[0],
                                                                 V1_max=V_max[0], H_loss=H_loss_b[0], A_num=A_hydro[0],
                                                                 is_print=cal_N_is_print)
    N_dianzhan_2, Q_out_2, Z_dianzhan_2, Z_dianzhan_2_24 = cal_N(station_num=1, Q1_fadian_set=Q_fadian_2,
                                                                 Q1_out=Q_out_1,
                                                                 Water1_data=water_data[1], Z1_begin=Z_up_begin[1],
                                                                 V1_max=V_max[1], H_loss=H_loss_b[1], A_num=A_hydro[1],
                                                                 is_print=cal_N_is_print)
    N_dianzhan_3, Q_out_3, Z_dianzhan_3, Z_dianzhan_3_24 = cal_N(station_num=2, Q1_fadian_set=Q_fadian_3,
                                                                 Q1_out=Q_out_2,
                                                                 Water1_data=water_data[2], Z1_begin=Z_up_begin[2],
                                                                 V1_max=V_max[2], H_loss=H_loss_b[2], A_num=A_hydro[2],
                                                                 is_print=cal_N_is_print)
    N_dianzhan_4, Q_out_4, Z_dianzhan_4, Z_dianzhan_4_24 = cal_N(station_num=3, Q1_fadian_set=Q_fadian_4,
                                                                 Q1_out=Q_out_0,
                                                                 Water1_data=water_data[3], Z1_begin=Z_up_begin[3],
                                                                 V1_max=V_max[3], H_loss=H_loss_b[3], A_num=A_hydro[3],
                                                                 is_print=cal_N_is_print)
    N_dianzhan_5, Q_out_5, Z_dianzhan_5, Z_dianzhan_5_24 = cal_N(station_num=4, Q1_fadian_set=Q_fadian_5,
                                                                 Q1_out=Q_out_4,
                                                                 Water1_data=water_data[4], Z1_begin=Z_up_begin[4],
                                                                 V1_max=V_max[4], H_loss=H_loss_b[4], A_num=A_hydro[4],
                                                                 is_print=cal_N_is_print)

    if is_print:
        print(f'Power output of station 1: {N_dianzhan_1}')
        print(f'Power output of station 2: {N_dianzhan_2}')
        print(f'Power output of station 3: {N_dianzhan_3}')
        print(f'Power output of station 4: {N_dianzhan_4}')
        print(f'Power output of station 5: {N_dianzhan_5}')
        N_dianzhan = N_dianzhan_1 + N_dianzhan_2 + N_dianzhan_3 + N_dianzhan_4 + N_dianzhan_5
        print(f'Total power output of 5 stations: {N_dianzhan}')
        print('-----------------------------------------------------------')
        print(f'Water level of station 1: {Z_dianzhan_1}')
        print(f'Water level of station 2: {Z_dianzhan_2}')
        print(f'Water level of station 3: {Z_dianzhan_3}')
        print(f'Water level of station 4: {Z_dianzhan_4}')
        print(f'Water level of station 5: {Z_dianzhan_5}')
    else:
        pass

    if is_N:
        if is_constant:
            if station_num == 0:
                return N_dianzhan_1
            elif station_num == 1:
                return N_dianzhan_2
            elif station_num == 2:
                return N_dianzhan_3
            elif station_num == 3:
                return N_dianzhan_4
            else:
                return N_dianzhan_5
        else:
            return N_dianzhan_1, N_dianzhan_2, N_dianzhan_3, N_dianzhan_4, N_dianzhan_5
    elif is_Z:
        if station_num == 0:
            return Z_dianzhan_1
        elif station_num == 1:
            return Z_dianzhan_2
        elif station_num == 2:
            return Z_dianzhan_3
        elif station_num == 3:
            return Z_dianzhan_4
        else:
            return Z_dianzhan_5
    else:
        if station_num == 0:
            return Z_dianzhan_1_24
        elif station_num == 1:
            return Z_dianzhan_2_24
        elif station_num == 2:
            return Z_dianzhan_3_24
        elif station_num == 3:
            return Z_dianzhan_4_24
        else:
            return Z_dianzhan_5_24


# Objective function
def demo_func3(x_set):
    # -------------------------------
    # Get generation flow data
    # -------------------------------
    N_dianzhan_1, N_dianzhan_2, N_dianzhan_3, N_dianzhan_4, N_dianzhan_5 = \
        cal_N_beta(x_set, is_constant=False, is_print=False, is_N=True, is_Z=False, cal_N_is_print=False)

    # -------------------------------
    # Calculate variables for objective function
    # -------------------------------
    # Calculate Price_gw
    # Grid electricity price (24-hour)
    Price_gw = np.array([
        0.3502, 0.3502, 0.3502, 0.3502, 0.3502, 0.3502, 0.3502, 0.3502,
        0.6589, 0.6589, 0.9602, 0.9602, 0.9602, 0.9602, 0.9602, 0.9602,
        0.6589, 0.6589, 0.6589, 0.6589, 0.6589, 1.1491, 1.1491, 0.6589
    ])

    # Calculate P_grid_state_gw (Grid purchase status)
    P_grid_state_gw = np.full(n_hours, np.nan)
    N_dianzhan_all = N_dianzhan_1 + N_dianzhan_2 + N_dianzhan_3 + N_dianzhan_4 + N_dianzhan_5
    for i in range(n_hours):
        if load_set[i] - N_dianzhan_all[i] > 0:
            temp = 1  # Need to purchase from grid
        else:
            temp = 0  # No need to purchase
        P_grid_state_gw[i] = temp

    # Calculate P_grid_gw (Grid purchase power)
    P_grid_gw = np.full(n_hours, np.nan)
    for i in range(n_hours):
        P_grid_gw[i] = round(load_set[i] - N_dianzhan_all[i], 2)

    # Calculate Sale_gw
    # Electricity selling price (24-hour)
    Sale_gw = np.array([
        0.307, 0.307, 0.307, 0.307, 0.307, 0.307, 0.307, 0.307,
        0.307, 0.307, 0.307, 0.307, 0.307, 0.307, 0.307, 0.307,
        0.307, 0.307, 0.307, 0.307, 0.307, 0.307, 0.307, 0.307
    ])

    # Calculate Price_gw_store
    Price_gw_store = 1.3  # Unit: Yuan/kW·d

    # Calculate y_gw (Grid connection flag)
    P_grid_gw_max_1 = max(P_grid_gw)
    y_gw = 0
    if P_grid_gw_max_1 > 0:
        y_gw = 1  # Grid connection needed

    # Calculate P_grid_gw_max (Max grid purchase power)
    P_grid_gw_max = P_grid_gw_max_1

    # Calculate objective function - minimize cost
    # All variables are assumed to be NumPy arrays (scalars will be broadcast automatically)
    obj_value = (
            np.sum(1000 * Price_gw * P_grid_state_gw * P_grid_gw) +  # Grid purchase cost
            np.sum(1000 * (1 - P_grid_state_gw) * Sale_gw * P_grid_gw) +  # Electricity sales income
            1000 * Price_gw_store * y_gw * P_grid_gw_max  # Grid connection cost
    )
    y = obj_value
    return y


# -------------------------------
# Constraint Conditions
# -------------------------------

# Power output constraints (min/max)
def constraint_N_min(x1, station_num1):
    P_hydro_min_new = np.ones(24) * P_hydro_min[station_num1]
    constraint = P_hydro_min_new - cal_N_beta(x1, station_num=station_num1, is_constant=True, is_print=False, is_N=True,
                                              is_Z=False, cal_N_is_print=False)
    return constraint


def constraint_N_max(x1, station_num1):
    P_hydro_max_new = np.ones(24) * P_hydro_max[station_num1]
    constraint = cal_N_beta(x1, station_num=station_num1, is_constant=True, is_print=False, is_N=True,
                            is_Z=False, cal_N_is_print=False) - P_hydro_max_new
    return constraint


# Water level constraints (min/max)
def constraint_Z_min(x1, station_num1):
    Z_up_min_new = np.ones(24) * Z_up_min[station_num1]
    constraint = Z_up_min_new - cal_N_beta(x1, station_num=station_num1, is_constant=True, is_print=False, is_N=False,
                                           is_Z=True, cal_N_is_print=False)
    return constraint


def constraint_Z_max(x1, station_num1):
    Z_up_max_new = np.ones(24) * Z_up_max[station_num1]
    constraint = cal_N_beta(x1, station_num=station_num1, is_constant=True, is_print=False, is_N=False,
                            is_Z=True, cal_N_is_print=False) - Z_up_max_new
    return constraint


# Final water level constraints
Z_up_end_range = np.array([0.1, 0.1, 0.1, 0.1, 0.1])


def constraint_Z_end_min(x1, station_num1):
    Z_up_end_min_new = Z_up_begin[station_num1] - Z_up_end_range[station_num1]
    constraint = Z_up_end_min_new - cal_N_beta(x1, station_num=station_num1, is_constant=True, is_print=False,
                                               is_N=False, is_Z=False, cal_N_is_print=False)
    return constraint


def constraint_Z_end_max(x1, station_num1):
    Z_up_end_max_new = Z_up_begin[station_num1] + Z_up_end_range[station_num1]
    constraint = cal_N_beta(x1, station_num=station_num1, is_constant=True, is_print=False, is_N=False,
                            is_Z=False, cal_N_is_print=False) - Z_up_end_max_new
    return constraint


# -------------------------------
# Power Flow Calculation
# -------------------------------

# Define test system data (IEEE 12-bus)
buses = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Bus
    [3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1],  # Type
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Vm
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Va
    [0.0, 0.0, 90, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45, 0.0],  # Pd
    [0.0, 0.0, 40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30, 0.0],  # Qd
    [0.0, 0.4, 1.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Pg
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Qg
])

lines = np.array([
    [1, 1, 2, 2, 4, 4, 5, 7, 8, 10, 11],  # From
    [4, 7, 3, 4, 5, 10, 6, 8, 9, 11, 12],  # To
    [0.000294, 0.001647, 0.05, 0.05, 0.000989, 0.000294, 0.000184, 0.003616, 0.014094, 0.004782, 0.009547],  # R
    [0.001031, 0.075368, 0.2, 0.12, 0.05451, 0.001031, 0.013522, 0.016546, 0.380702, 0.152801, 0.223636],  # X
    [0, 0, 0.02, 0.01, 0, 0, 0, 0.00308, 0, 0, 0]  # B
])

# Calculate PD_set and QD_set (12 x 24)
baseMVA = 100.0
P_scenes = load_set
PD = np.zeros((bus_num, n_hours), dtype=float)
QD = np.zeros((bus_num, n_hours), dtype=float)
PD_ori = buses[4, :] / baseMVA
QD_ori = buses[5, :] / baseMVA
P_sum = P_scenes / baseMVA

# 24-hour load data
Q_factor = QD_ori / sum(QD_ori)
Q_factor = Q_factor.reshape(12, 1)
P_factor = PD_ori / sum(PD_ori)
P_factor = P_factor.reshape(12, 1)

QD_set = Q_factor * sum(QD_ori) * P_sum / sum(PD_ori)
PD_set = P_factor * P_sum

# Record 24-hour measurements
v_i_set = np.zeros(24)
v_a_set = np.zeros(24)
P_ij_set = np.zeros(24)
Q_ij_set = np.zeros(24)

# Power flow calculation loop
for t in range(T):
    power_flow_result = run_power_flow(buses, lines, max_iter=100, tolerance=1e-6, verbose=True)
    # Record constraint values
    v_i_set[t] = power_flow_result[1]
    v_a_set[t] = power_flow_result[2]
    P_ij_set[t] = power_flow_result[3]
    Q_ij_set[t] = power_flow_result[4]
    # Update bus data
    buses[2] = power_flow_result[1]
    buses[3] = power_flow_result[2]
    buses[4] = PD_set.loc[t]
    buses[5] = QD_set.loc[t]
    buses[6] = power_flow_result[3]
    buses[7] = power_flow_result[4]

# Power flow constraint limits
v_i_min = 0.9
v_i_max = 1.1
v_a_min = -2 * np.pi
v_a_max = 2 * np.pi
P_ij_min = np.array([-125, -137.13, -130, -70, -144.52, -40, -633.62, -165.22, -45.5, -100.71, -54.86])
P_ij_max = np.array([125, 137.13, 130, 70, 144.52, 40, 633.62, 165.22, 45.5, 100.71, 54.86])
Q_ij_min = np.array([-125, -137.13, -130, -70, -144.52, -40, -633.62, -165.22, -45.5, -100.71, -54.86])
Q_ij_max = np.array([125, 137.13, 130, 70, 144.52, 40, 633.62, 165.22, 45.5, 100.71, 54.86])

# Expand to 24-hour format
v_i_min_new = np.ones(24) * v_i_min
v_i_max_new = np.ones(24) * v_i_max
v_a_min_new = np.ones(24) * v_a_min
v_a_max_new = np.ones(24) * v_a_max
P_ij_min_new = np.ones(24) * P_ij_min
P_ij_max_new = np.ones(24) * P_ij_max
Q_ij_min_new = np.ones(24) * Q_ij_min
Q_ij_max_new = np.ones(24) * Q_ij_max

# Voltage limits constraints
constraint_1 = v_i_min_new - v_i_set
constraint_2 = v_i_set - v_i_max_new
# Phase angle limits constraints
constraint_3 = v_a_min_new - v_a_set
constraint_4 = v_a_set - v_a_max_new
# Branch active power limits constraints
constraint_5 = P_ij_min_new - P_ij_set
constraint_6 = P_ij_set - P_ij_max_new
# Branch reactive power limits constraints
constraint_7 = Q_ij_min_new - Q_ij_set
constraint_8 = Q_ij_set - Q_ij_max_new

# -------------------------------
# Constraint Integration
# -------------------------------
constraint_ueq = ()
for station in range(n_stations):  # Iterate all stations (0-4)
    constraint_ueq += (
        # Power output constraints
        lambda x, station=station: constraint_N_min(x, station_num1=station),
        lambda x, station=station: constraint_N_max(x, station_num1=station),
        # Water level constraints
        lambda x, station=station: constraint_Z_min(x, station_num1=station),
        lambda x, station=station: constraint_Z_max(x, station_num1=station),
        # Final water level constraints
        lambda x, station=station: constraint_Z_end_min(x, station_num1=station),
        lambda x, station=station: constraint_Z_end_max(x, station_num1=station),
    )

# Add power flow constraints
constraint_ueq += (
    lambda: constraint_1,
    lambda: constraint_2,
    lambda: constraint_3,
    lambda: constraint_4,
    lambda: constraint_5,
    lambda: constraint_6,
    lambda: constraint_7,
    lambda: constraint_8
)

# PSO algorithm parameters
dim_self = n_hours * n_stations

# Generate bounds arrays
lb_self = np.concatenate([[0] * n_hours] * 5)  # Lower bounds
ub_self = np.concatenate([  # Upper bounds
    [10.6] * n_hours,  # Station 0
    [75.694] * n_hours,  # Station 1
    [64.66] * n_hours,  # Station 2
    [5.2] * n_hours,  # Station 3
    [7.6] * n_hours  # Station 4
])

# generate one-dimensional load standard array
load_set_max = max(load_set)
load_set_biaozhun = np.zeros(n_hours, dtype=float)
for i in range(n_hours):
    load_set_biaozhun[i] = round((load_set[i] / load_set_max), 2)

# Initialize PSO
pso = PSO(func=demo_func3, dim=dim_self, pop=100, max_iter=100,
          lb=lb_self, ub=ub_self, dianzhan=5,
          load_set_standard=load_set_biaozhun, w=0.8, c1=0.5, c2=0.5,
          constraint_ueq=constraint_ueq, verbose=True, fr=0.2, v_num=0.25)
pso.run()


# -------------------------------
# Print and Output Results
# -------------------------------

print('')
print('------------------------------------- Final Results -----------------------------------')
new_gbest_x = pso.gbest_x.flatten()
N_dianzhan_1_last, N_dianzhan_2_last, N_dianzhan_3_last, N_dianzhan_4_last, N_dianzhan_5_last = \
    cal_N_beta(new_gbest_x, is_constant=False, is_print=True, is_N=True, is_Z=False, cal_N_is_print=False)
print('best_y is', pso.gbest_y)
print('')

print('Particle Swarm Optimization Output Power Scheme:')
N_dianzhan_1_last1, N_dianzhan_2_last1, N_dianzhan_3_last1, N_dianzhan_4_last1, N_dianzhan_5_last1 = cal_N_beta(
    new_gbest_x, is_constant=False, is_print=True, is_N=True, is_Z=False, cal_N_is_print=True)

N_dianzhan_last1 = N_dianzhan_1_last1 + N_dianzhan_2_last1 + N_dianzhan_3_last1 + N_dianzhan_4_last1 + N_dianzhan_5_last1

# Specify output txt file path
file_name = r"C:\Users\Longwen-Liu\Desktop\其他智能算法\粒子群算法\粒子群算法输出出力方案.txt"
# Open file in write mode
with open(file_name, "w") as file:
    # Write each element to a new line in the file
    for num in N_dianzhan_last1:
        file.write(str(num) + "\n")

# Plotting
plt.figure(figsize=(12, 8))
gbest_y_hist = np.array(pso.gbest_y_hist, dtype=object)
x_num = list(range(len(gbest_y_hist)))
plt.scatter(x_num, gbest_y_hist)
plt.plot(gbest_y_hist)

# Set x-axis range from 0 to max iterations
plt.xlim(-1, len(gbest_y_hist))
# Disable scientific notation on y-axis
plt.ticklabel_format(style='plain', axis='y')
plt.show()
