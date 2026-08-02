#version 3.7;
global_settings { assumed_gamma 1.0 }
background { color rgbt <1,1,1,1> }
camera { orthographic location <9.543885,6.639224,-22.407381> look_at <0,0,0> right x*14.274331830212367 up y*10.705748872659274 }
light_source { <-8.299030,16.598060,-16.598060> color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }
light_source { <16.598060,-8.299030,-8.299030> color rgb <0.35,0.40,0.52> }
plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }
cylinder { <-2.616424,-1.629161,-1.704521>, <-3.528355,-1.983332,-1.270027>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.616424,-1.629161,-1.704521>, <-2.465904,-0.597033,-1.445111>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.616424,-1.629161,-1.704521>, <-2.745506,-1.676140,-2.767687>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.616424,-1.629161,-1.704521>, <-1.547482,-2.495619,-1.235973>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.547482,-2.495619,-1.235973>, <-1.809359,-3.503457,-0.578902>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.547482,-2.495619,-1.235973>, <-0.252762,-2.163794,-1.522738>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.252762,-2.163794,-1.522738>, <0.396764,-2.840130,-1.145468>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.252762,-2.163794,-1.522738>, <0.235277,-1.088235,-2.377110>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.235277,-1.088235,-2.377110>, <-0.542025,-0.833914,-3.096007>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.235277,-1.088235,-2.377110>, <1.475158,-1.514679,-3.150920>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.235277,-1.088235,-2.377110>, <0.428763,0.230027,-1.596547>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.475158,-1.514679,-3.150920>, <1.287648,-2.423109,-3.706794>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.475158,-1.514679,-3.150920>, <2.322141,-1.676779,-2.504638>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.475158,-1.514679,-3.150920>, <1.741399,-0.735842,-3.844939>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.428763,0.230027,-1.596547>, <0.134005,1.286172,-2.166292>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.428763,0.230027,-1.596547>, <0.891454,0.310227,-0.292077>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.891454,0.310227,-0.292077>, <0.993473,1.285310,-0.043026>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.891454,0.310227,-0.292077>, <1.411286,-0.617467,0.729190>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.411286,-0.617467,0.729190>, <1.960126,0.042562,1.410654>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.411286,-0.617467,0.729190>, <2.398683,-1.672837,0.315482>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.411286,-0.617467,0.729190>, <0.273104,-1.185377,1.607547>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.398683,-1.672837,0.315482>, <2.743888,-2.167403,1.204670>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.398683,-1.672837,0.315482>, <1.992558,-2.435772,-0.313802>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.398683,-1.672837,0.315482>, <3.237422,-1.227504,-0.184427>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.273104,-1.185377,1.607547>, <0.314750,-2.370363,1.944496>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.273104,-1.185377,1.607547>, <-0.876299,-0.450303,1.858156>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.876299,-0.450303,1.858156>, <-1.518444,-1.037612,2.371169>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.876299,-0.450303,1.858156>, <-1.177059,0.995356,2.057511>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.177059,0.995356,2.057511>, <-1.682560,1.032977,3.022790>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.177059,0.995356,2.057511>, <-2.118446,1.477180,0.973613>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.177059,0.995356,2.057511>, <0.046454,1.912248,2.348480>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.118446,1.477180,0.973613>, <-1.648790,1.479517,-0.004194>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.118446,1.477180,0.973613>, <-2.478178,2.462085,1.218379>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.118446,1.477180,0.973613>, <-2.971665,0.813982,0.928738>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.046454,1.912248,2.348480>, <0.860800,1.694211,3.243328>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.046454,1.912248,2.348480>, <0.158172,3.008193,1.498196>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.158172,3.008193,1.498196>, <-0.529402,3.203098,0.775500>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.158172,3.008193,1.498196>, <1.095331,4.049784,1.794815>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.095331,4.049784,1.794815>, <0.861531,4.596927,2.715251>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.095331,4.049784,1.794815>, <2.102333,3.650435,1.937981>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.095331,4.049784,1.794815>, <1.146141,4.795574,0.995252>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
sphere { <-2.616424,-1.629161,-1.704521>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-3.528355,-1.983332,-1.270027>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.465904,-0.597033,-1.445111>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.745506,-1.676140,-2.767687>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.547482,-2.495619,-1.235973>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.809359,-3.503457,-0.578902>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.252762,-2.163794,-1.522738>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.396764,-2.840130,-1.145468>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.235277,-1.088235,-2.377110>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.542025,-0.833914,-3.096007>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.475158,-1.514679,-3.150920>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.287648,-2.423109,-3.706794>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.322141,-1.676779,-2.504638>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.741399,-0.735842,-3.844939>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.428763,0.230027,-1.596547>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.134005,1.286172,-2.166292>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.891454,0.310227,-0.292077>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.993473,1.285310,-0.043026>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.411286,-0.617467,0.729190>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.960126,0.042562,1.410654>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.398683,-1.672837,0.315482>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.743888,-2.167403,1.204670>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.992558,-2.435772,-0.313802>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <3.237422,-1.227504,-0.184427>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.273104,-1.185377,1.607547>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.314750,-2.370363,1.944496>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.876299,-0.450303,1.858156>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.518444,-1.037612,2.371169>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.177059,0.995356,2.057511>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.682560,1.032977,3.022790>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.118446,1.477180,0.973613>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.648790,1.479517,-0.004194>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.478178,2.462085,1.218379>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.971665,0.813982,0.928738>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.046454,1.912248,2.348480>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.860800,1.694211,3.243328>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.158172,3.008193,1.498196>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.529402,3.203098,0.775500>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.095331,4.049784,1.794815>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.861531,4.596927,2.715251>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.102333,3.650435,1.937981>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.146141,4.795574,0.995252>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
cylinder { <-2.950156,-1.871903,-1.626426>, <-2.828875,-1.783689,-1.654806>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.828875,-1.783689,-1.654806>, 0.12, <-3.003687,-1.910839,-1.613899>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-3.582991,-2.190626,-0.908856>, <-3.571444,-2.146816,-0.985187>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-3.571444,-2.146816,-0.985187>, 0.12, <-3.600063,-2.255399,-0.796003>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.349903,-0.198639,-1.380100>, <-2.307518,-0.053074,-1.356347>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.307518,-0.053074,-1.356347>, 0.12, <-2.246755,0.155608,-1.322294>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.403712,-1.580991,-2.992461>, <-2.439591,-1.590979,-2.968866>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.439591,-1.590979,-2.968866>, 0.12, <-2.260556,-1.541139,-3.086605>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.241258,-2.218957,-1.157965>, <-1.283272,-2.256915,-1.168668>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.283272,-2.256915,-1.168668>, 0.12, <-1.122869,-2.111997,-1.127807>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.830564,-3.238073,-0.254061>, <-1.826095,-3.294006,-0.322526>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.826095,-3.294006,-0.322526>, 0.12, <-1.837202,-3.154996,-0.152371>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.125264,-2.142178,-1.340999>, <0.130657,-2.141870,-1.338406>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.130657,-2.141870,-1.338406>, 0.12, <0.328670,-2.130547,-1.243210>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.221649,-3.216220,-1.079962>, <0.241919,-3.172685,-1.087545>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.241919,-3.172685,-1.087545>, 0.12, <0.150193,-3.369685,-1.053232>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.088145,-1.069737,-2.109797>, <-0.552985,-1.043151,-1.725599>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.552985,-1.043151,-1.725599>, 0.12, <-0.722396,-1.033461,-1.585578>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.146408,-0.964718,-3.148706>, <-0.270192,-0.923791,-3.132217>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.270192,-0.923791,-3.132217>, 0.12, <-0.062964,-0.992307,-3.159821>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.230934,-1.457112,-2.814111>, <1.272816,-1.466985,-2.871870>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.272816,-1.466985,-2.871870>, 0.12, <1.144889,-1.436831,-2.695446>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.083505,-2.719634,-3.923123>, <1.099916,-2.695796,-3.905732>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.099916,-2.695796,-3.905732>, 0.12, <0.992984,-2.851119,-4.019047>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.678994,-1.734591,-2.290830>, <2.719806,-1.741203,-2.266378>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.719806,-1.741203,-2.266378>, 0.12, <2.906728,-1.771485,-2.154383>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.828496,-0.458233,-4.147838>, <1.853606,-0.378200,-4.235161>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.853606,-0.378200,-4.235161>, 0.12, <1.899228,-0.232786,-4.393822>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.497935,-0.160210,-1.735580>, <0.484801,-0.086115,-1.709181>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.484801,-0.086115,-1.709181>, 0.12, <0.521034,-0.290525,-1.782008>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.238221,1.378252,-1.994917>, <-0.193163,1.367106,-2.015662>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.193163,1.367106,-2.015662>, 0.12, <-0.388139,1.415338,-1.925894>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.124133,0.025237,-0.494663>, <1.274515,-0.158955,-0.625597>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.274515,-0.158955,-0.625597>, 0.12, <1.396394,-0.308235,-0.731713>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.005199,1.031414,0.291338>, <1.001653,1.108200,0.190216>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.001653,1.108200,0.190216>, 0.12, <1.007795,0.975207,0.365359>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.210357,-0.270196,0.853408>, <0.669762,0.664130,1.187612>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.669762,0.664130,1.187612>, 0.12, <0.564514,0.846034,1.252678>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.315222,0.162224,1.600358>, <2.258217,0.143014,1.569904>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.258217,0.143014,1.569904>, 0.12, <2.444220,0.205695,1.669272>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.229774,-1.638833,0.698514>, <2.264079,-1.645739,0.620721>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.264079,-1.645739,0.620721>, 0.12, <2.175603,-1.627928,0.821357>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.885610,-2.217703,1.596824>, <2.924835,-2.231624,1.705363>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.924835,-2.231624,1.705363>, 0.12, <2.999071,-2.257972,1.910777>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.741828,-2.696892,-0.526760>, <1.651661,-2.790796,-0.603345>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.651661,-2.790796,-0.603345>, 0.12, <1.520326,-2.927573,-0.714895>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <3.578553,-1.067722,-0.370166>, <3.778767,-0.973944,-0.479178>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <3.778767,-0.973944,-0.479178>, 0.12, <3.957455,-0.890249,-0.576470>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.141627,-1.328482,1.235210>, <-0.031661,-1.517097,0.744464>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.031661,-1.517097,0.744464>, 0.12, <-0.100529,-1.592057,0.549431>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.593741,-2.676368,2.014676>, <0.550285,-2.628704,2.003744>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.550285,-2.628704,2.003744>, 0.12, <0.696423,-2.788993,2.040505>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.575624,-0.162485,1.914327>, <-0.374177,0.030349,1.951961>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.374177,0.030349,1.951961>, 0.12, <-0.216681,0.181111,1.981384>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.840828,-1.282212,2.258743>, <-1.759493,-1.220502,2.287107>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.759493,-1.220502,2.287107>, 0.12, <-1.928361,-1.348626,2.228218>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.889403,0.722246,2.195588>, <-0.560540,0.410011,2.353446>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.560540,0.410011,2.353446>, 0.12, <-0.409863,0.266953,2.425772>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.099584,0.983239,3.026929>, <-2.086334,0.984819,3.026797>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.086334,0.984819,3.026797>, 0.12, <-2.304775,0.958766,3.028966>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.784431,1.266645,1.116825>, <-1.859448,1.313929,1.084661>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.859448,1.313929,1.084661>, 0.12, <-1.684488,1.203649,1.159677>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.451070,1.413901,-0.368887>, <-1.429835,1.406855,-0.408053>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.429835,1.406855,-0.408053>, 0.12, <-1.326268,1.372484,-0.599083>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.679584,2.828294,1.176827>, <-2.766449,2.986236,1.158907>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.766449,2.986236,1.158907>, 0.12, <-2.871948,3.178060,1.137142>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-3.303210,0.591481,0.798461>, <-3.331106,0.572760,0.787500>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-3.331106,0.572760,0.787500>, 0.12, <-3.504772,0.456212,0.719260>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.093070,1.530072,2.244200>, <-0.177642,1.298418,2.180992>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.177642,1.298418,2.180992>, 0.12, <-0.250726,1.098230,2.126369>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.673935,1.664191,2.868388>, <0.631675,1.657402,2.783595>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.631675,1.657402,2.783595>, 0.12, <0.533793,1.641677,2.587198>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.334250,3.337197,1.305445>, <0.396983,3.454414,1.236773>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.396983,3.454414,1.236773>, 0.12, <0.489214,3.626749,1.135809>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.294579,3.053819,1.090102>, <-0.289254,3.050434,1.097236>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.289254,3.050434,1.097236>, 0.12, <-0.166252,2.972240,1.262027>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.884959,3.691904,1.731058>, <0.860953,3.651066,1.723782>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.860953,3.651066,1.723782>, 0.12, <0.750759,3.463604,1.690386>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.669251,4.391151,3.026835>, <0.733463,4.459870,2.922781>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.733463,4.459870,2.922781>, 0.12, <0.632745,4.352082,3.085992>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.339013,3.311032,2.010009>, <2.303761,3.361583,1.999281>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.303761,3.361583,1.999281>, 0.12, <2.427736,3.183801,2.037010>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.213094,5.085043,0.698394>, <1.199032,5.024245,0.760745>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.199032,5.024245,0.760745>, 0.12, <1.234102,5.175872,0.605248>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
