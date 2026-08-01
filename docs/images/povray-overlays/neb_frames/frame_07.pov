#version 3.7;
global_settings { assumed_gamma 1.0 }
background { color rgbt <1,1,1,1> }
camera { orthographic location <5.502052,3.827514,-12.917860> look_at <0,0,0> right x*8.229155306400001 up y*6.1718664798 }
light_source { <-4.784393,9.568785,-9.568785> color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }
light_source { <9.568785,-4.784393,-4.784393> color rgb <0.35,0.40,0.52> }
plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }
cylinder { <-1.102788,0.420173,0.143212>, <0.408859,-0.193135,-0.081722>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.102788,0.420173,0.143212>, <-1.119400,1.039355,0.960399>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.102788,0.420173,0.143212>, <-1.800046,-0.316877,0.295349>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.102788,0.420173,0.143212>, <-1.400012,0.970316,-0.669701>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.408859,-0.193135,-0.081722>, <0.640385,-0.756851,0.823046>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.408859,-0.193135,-0.081722>, <0.334685,-0.816812,-0.973508>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.408859,-0.193135,-0.081722>, <1.053970,0.675170,-0.216186>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
sphere { <2.984346,-1.021340,-0.280889>, 0.5304 texture { pigment { color rgb <0.200000,0.720000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.102788,0.420173,0.143212>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.408859,-0.193135,-0.081722>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.119400,1.039355,0.960399>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.800046,-0.316877,0.295349>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.400012,0.970316,-0.669701>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.640385,-0.756851,0.823046>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.334685,-0.816812,-0.973508>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.053970,0.675170,-0.216186>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
cylinder { <3.258469,-1.170719,0.000078>, <3.635805,-1.376344,0.386836>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <3.635805,-1.376344,0.386836>, 0.12, <3.779393,-1.454590,0.534009>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.757510,0.300391,-0.063755>, <-0.518818,0.217586,-0.206831>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.518818,0.217586,-0.206831>, 0.12, <-0.337957,0.154843,-0.315242>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.018786,-0.041899,-0.118746>, <-0.562389,0.183431,-0.173909>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.562389,0.183431,-0.173909>, 0.12, <-0.766713,0.262650,-0.193303>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.000816,0.645602,0.874980>, <-0.899923,0.310589,0.802304>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.899923,0.310589,0.802304>, 0.12, <-0.837808,0.104337,0.757561>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.391634,-0.251392,0.222473>, <-0.934938,-0.178166,0.140981>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.934938,-0.178166,0.140981>, 0.12, <-0.721008,-0.143864,0.102807>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.542683,1.028465,-0.278979>, <-1.837529,1.148637,0.528490>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.837529,1.148637,0.528490>, 0.12, <-1.912261,1.179097,0.733154>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.362968,-0.659922,0.522971>, <-0.383415,-0.399136,-0.284372>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.383415,-0.399136,-0.284372>, 0.12, <-0.528728,-0.348364,-0.441554>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.505120,-0.439194,-0.904538>, <0.682626,-0.045911,-0.832706>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.682626,-0.045911,-0.832706>, 0.12, <0.771902,0.151889,-0.796579>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.058809,0.268266,-0.320134>, <1.062156,-0.013258,-0.392053>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.062156,-0.013258,-0.392053>, 0.12, <1.064690,-0.226399,-0.446502>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.102788,0.420173,-0.206788>, <0.408859,-0.193135,-0.431722>, 0.04 pigment { color rgb <0.48,0.08,0.72> } finish { emission 0.25 } }
sphere { <-0.346964,0.113519,-0.319255>, 0.075 pigment { color rgb <0.48,0.08,0.72> } }
