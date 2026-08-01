#version 3.7;
global_settings { assumed_gamma 1.0 }
background { color rgbt <1,1,1,1> }
camera { orthographic location <11.335126,7.885305,-26.612905> look_at <0,0,0> right x*16.953405983667366 up y*12.715054487750525 }
light_source { <-9.856631,19.713263,-19.713263> color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }
light_source { <19.713263,-9.856631,-9.856631> color rgb <0.35,0.40,0.52> }
plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }
cylinder { <-2.553934,-0.629269,-2.432156>, <-3.439988,-0.968161,-1.937063>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.553934,-0.629269,-2.432156>, <-2.277172,0.335749,-2.076293>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.553934,-0.629269,-2.432156>, <-2.785434,-0.534473,-3.482627>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.553934,-0.629269,-2.432156>, <-1.502907,-1.576882,-2.101811>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.502907,-1.576882,-2.101811>, <-1.778177,-2.626382,-1.500122>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.502907,-1.576882,-2.101811>, <-0.208213,-1.302067,-2.436876>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.208213,-1.302067,-2.436876>, <0.454462,-1.899730,-1.926241>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.208213,-1.302067,-2.436876>, <0.348792,-0.103405,-3.055149>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.348792,-0.103405,-3.055149>, <-0.338465,0.215163,-3.832453>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.348792,-0.103405,-3.055149>, <1.679572,-0.506442,-3.689028>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.348792,-0.103405,-3.055149>, <0.480683,1.164689,-2.178391>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.679572,-0.506442,-3.689028>, <1.544356,-1.397809,-4.291786>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.679572,-0.506442,-3.689028>, <2.430508,-0.712700,-2.931603>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.679572,-0.506442,-3.689028>, <2.037207,0.285153,-4.328207>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.480683,1.164689,-2.178391>, <0.266939,2.239972,-2.744472>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.480683,1.164689,-2.178391>, <0.934964,1.233714,-0.865227>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.934964,1.233714,-0.865227>, <1.000797,2.219244,-0.635635>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.934964,1.233714,-0.865227>, <1.434894,0.393878,0.262793>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.434894,0.393878,0.262793>, <1.109903,0.918396,1.163365>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.434894,0.393878,0.262793>, <2.952668,0.343365,0.193768>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.434894,0.393878,0.262793>, <0.864423,-1.032021,0.276832>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.952668,0.343365,0.193768>, <3.329267,-0.300958,0.969797>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.952668,0.343365,0.193768>, <3.267314,-0.033228,-0.773725>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.952668,0.343365,0.193768>, <3.362227,1.332095,0.340522>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.864423,-1.032021,0.276832>, <1.447092,-1.923241,-0.364728>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.864423,-1.032021,0.276832>, <-0.291522,-1.348786,0.944153>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.291522,-1.348786,0.944153>, <-0.604956,-2.276146,0.686185>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.291522,-1.348786,0.944153>, <-1.084430,-0.757097,2.036491>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.084430,-0.757097,2.036491>, <-1.458494,-1.629194,2.569653>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.084430,-0.757097,2.036491>, <-2.255545,0.043985,1.503627>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.084430,-0.757097,2.036491>, <-0.217201,-0.073684,3.120318>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.255545,0.043985,1.503627>, <-1.919598,0.938324,0.996578>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.255545,0.043985,1.503627>, <-2.919811,0.308875,2.312834>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.255545,0.043985,1.503627>, <-2.809771,-0.562941,0.795168>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.217201,-0.073684,3.120318>, <0.747686,-0.620203,3.648150>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.217201,-0.073684,3.120318>, <-0.619837,1.202701,3.484377>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.619837,1.202701,3.484377>, <-1.419359,1.668295,3.072032>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.619837,1.202701,3.484377>, <0.046231,1.857894,4.568002>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.046231,1.857894,4.568002>, <-0.110111,1.355858,5.528425>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.046231,1.857894,4.568002>, <1.127969,1.860967,4.415961>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.046231,1.857894,4.568002>, <-0.273029,2.896501,4.694564>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
sphere { <-2.553934,-0.629269,-2.432156>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-3.439988,-0.968161,-1.937063>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.277172,0.335749,-2.076293>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.785434,-0.534473,-3.482627>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.502907,-1.576882,-2.101811>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.778177,-2.626382,-1.500122>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.208213,-1.302067,-2.436876>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.454462,-1.899730,-1.926241>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.348792,-0.103405,-3.055149>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.338465,0.215163,-3.832453>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.679572,-0.506442,-3.689028>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.544356,-1.397809,-4.291786>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.430508,-0.712700,-2.931603>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.037207,0.285153,-4.328207>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.480683,1.164689,-2.178391>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.266939,2.239972,-2.744472>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.934964,1.233714,-0.865227>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.000797,2.219244,-0.635635>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.434894,0.393878,0.262793>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.109903,0.918396,1.163365>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.952668,0.343365,0.193768>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <3.329267,-0.300958,0.969797>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <3.267314,-0.033228,-0.773725>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <3.362227,1.332095,0.340522>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.864423,-1.032021,0.276832>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.447092,-1.923241,-0.364728>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.291522,-1.348786,0.944153>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.604956,-2.276146,0.686185>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.084430,-0.757097,2.036491>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.458494,-1.629194,2.569653>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.255545,0.043985,1.503627>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.919598,0.938324,0.996578>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.919811,0.308875,2.312834>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.809771,-0.562941,0.795168>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.217201,-0.073684,3.120318>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.747686,-0.620203,3.648150>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.619837,1.202701,3.484377>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.419359,1.668295,3.072032>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.046231,1.857894,4.568002>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.110111,1.355858,5.528425>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.127969,1.860967,4.415961>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.273029,2.896501,4.694564>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
cylinder { <-2.340697,-0.990412,-2.454641>, <-2.269110,-1.111653,-2.462189>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.269110,-1.111653,-2.462189>, 0.12, <-2.157415,-1.300823,-2.473967>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-3.746960,-1.140523,-1.708022>, <-3.783710,-1.161158,-1.680602>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-3.783710,-1.161158,-1.680602>, 0.12, <-3.944505,-1.251443,-1.560628>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.264539,0.754870,-2.052259>, <-2.236929,1.670897,-1.999729>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.236929,1.670897,-1.999729>, 0.12, <-2.230312,1.890437,-1.987139>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.462085,-0.653138,-3.242286>, <-2.162020,-0.763257,-3.019253>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.162020,-0.763257,-3.019253>, 0.12, <-1.992647,-0.825415,-2.893361>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.322287,-1.668923,-2.469649>, <-1.052923,-1.806187,-3.018214>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.052923,-1.806187,-3.018214>, 0.12, <-0.958313,-1.854399,-3.210891>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.786378,-2.384860,-1.156611>, <-1.793338,-2.179888,-0.865084>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.793338,-2.179888,-0.865084>, 0.12, <-1.797633,-2.053376,-0.685150>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.131290,-1.113346,-2.069633>, <-0.114253,-1.071548,-1.988295>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.114253,-1.071548,-1.988295>, 0.12, <-0.073960,-0.972694,-1.795930>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.646381,-1.823870,-1.560437>, <0.686756,-1.807910,-1.483479>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.686756,-1.807910,-1.483479>, 0.12, <0.787285,-1.768174,-1.291868>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.006324,-0.236824,-2.851887>, <-0.915076,-0.595785,-2.305015>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.915076,-0.595785,-2.305015>, 0.12, <-1.094463,-0.665671,-2.198544>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.070387,0.400390,-4.097453>, <-0.075713,0.396711,-4.092188>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.075713,0.396711,-4.092188>, 0.12, <0.064708,0.493734,-4.230998>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.365899,-0.375976,-3.442071>, <1.346937,-0.368089,-3.427142>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.346937,-0.368089,-3.427142>, 0.12, <1.182633,-0.299750,-3.297783>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.524027,-1.596029,-4.661510>, <1.525584,-1.580846,-4.633190>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.525584,-1.580846,-4.633190>, 0.12, <1.514936,-1.684675,-4.826854>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.597395,-0.636152,-2.553860>, <2.671318,-0.602245,-2.386538>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.671318,-0.602245,-2.386538>, 0.12, <2.758734,-0.562148,-2.188673>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.258818,0.563359,-4.551568>, <2.482093,0.843654,-4.776607>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.482093,0.843654,-4.776607>, 0.12, <2.598174,0.989381,-4.893606>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.868335,1.064784,-2.305452>, <1.450144,0.914840,-2.496151>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.450144,0.914840,-2.496151>, 0.12, <1.653200,0.862509,-2.562707>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.097124,2.031403,-2.725572>, <-0.342427,1.890870,-2.712837>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.342427,1.890870,-2.712837>, 0.12, <-0.533126,1.781620,-2.702936>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.615788,1.007418,-0.712528>, <-0.157985,0.458814,-0.342344>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.157985,0.458814,-0.342344>, 0.12, <-0.325172,0.340278,-0.262359>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.606551,2.098303,-0.555990>, <0.661812,2.115255,-0.567154>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.661812,2.115255,-0.567154>, 0.12, <0.455302,2.051905,-0.525435>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.195665,0.460419,-0.075943>, <0.651048,0.611905,-0.847096>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.651048,0.611905,-0.847096>, 0.12, <0.525738,0.646760,-1.024529>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.211357,0.951367,1.569591>, <1.243819,0.961916,1.699572>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.243819,0.961916,1.699572>, 0.12, <1.296962,0.979187,1.912357>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.570826,0.176197,0.245254>, <2.517517,0.152859,0.252442>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.517517,0.152859,0.252442>, 0.12, <2.317504,0.065294,0.279411>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <3.512178,-0.551890,1.252599>, <3.684944,-0.788906,1.519717>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <3.684944,-0.788906,1.519717>, 0.12, <3.780754,-0.920346,1.667851>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <3.226640,-0.288149,-1.105027>, <3.210091,-0.391869,-1.239824>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <3.210091,-0.391869,-1.239824>, 0.12, <3.188785,-0.525399,-1.413363>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <3.649290,1.622157,0.439814>, <3.857417,1.832460,0.511804>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <3.857417,1.832460,0.511804>, 0.12, <4.007783,1.984397,0.563814>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.748980,-0.848555,0.636572>, <0.508495,-0.466369,1.385961>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.508495,-0.466369,1.385961>, 0.12, <0.448025,-0.370268,1.574396>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.657663,-2.065888,-0.698961>, <1.845784,-2.193327,-0.997560>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.845784,-2.193327,-0.997560>, 0.12, <1.956083,-2.268047,-1.172634>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.171889,-0.976914,0.789884>, <-0.120705,-0.817811,0.723881>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.120705,-0.817811,0.723881>, 0.12, <-0.058040,-0.623021,0.643073>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.939878,-2.433631,0.487626>, <-0.966092,-2.445958,0.472084>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.966092,-2.445958,0.472084>, 0.12, <-1.141527,-2.528450,0.368077>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.691784,-0.906158,2.039650>, <0.220311,-1.252417,2.046989>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.220311,-1.252417,2.046989>, 0.12, <0.425982,-1.330497,2.048644>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.867418,-1.706960,2.625634>, <-2.009110,-1.733905,2.645031>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.009110,-1.733905,2.645031>, 0.12, <-2.223308,-1.774640,2.674354>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.849266,-0.061359,1.519137>, <-1.689932,-0.102672,1.525220>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.689932,-0.102672,1.525220>, 0.12, <-1.477120,-0.157853,1.533344>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.775181,1.243870,0.747207>, <-1.672912,1.460242,0.570614>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.672912,1.460242,0.570614>, 0.12, <-1.597266,1.620290,0.439992>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-3.247703,0.465376,2.523534>, <-3.503342,0.587392,2.687805>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-3.503342,0.587392,2.687805>, 0.12, <-3.675095,0.669369,2.798171>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.986863,-0.734799,0.455310>, <-3.096213,-0.840918,0.245454>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-3.096213,-0.840918,0.245454>, 0.12, <-3.188976,-0.930939,0.067433>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.233553,-0.316735,2.778179>, <-0.250468,-0.568166,2.424241>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.250468,-0.568166,2.424241>, 0.12, <-0.259033,-0.695478,2.245026>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.557266,-0.404699,3.342048>, <0.407894,-0.235652,3.101934>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.407894,-0.235652,3.101934>, 0.12, <0.308150,-0.122769,2.941595>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.558677,1.618224,3.484271>, <-0.522501,1.864005,3.484208>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.522501,1.864005,3.484208>, 0.12, <-0.490465,2.081660,3.484152>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.141620,1.412599,3.256100>, <-1.017962,1.298754,3.338053>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.017962,1.298754,3.338053>, 0.12, <-0.872480,1.164818,3.434469>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.100945,1.663689,4.225915>, <-0.186426,1.550894,4.027227>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.186426,1.550894,4.027227>, 0.12, <-0.263518,1.449167,3.848039>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.153174,0.974312,5.698626>, <-0.142262,1.070991,5.655499>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.142262,1.070991,5.655499>, 0.12, <-0.164820,0.871134,5.744652>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.486469,1.680220,4.292640>, <1.474900,1.686053,4.296619>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.474900,1.686053,4.296619>, 0.12, <1.662685,1.591376,4.232022>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.357862,3.295487,4.794631>, <-0.361506,3.312621,4.798929>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.361506,3.312621,4.798929>, 0.12, <-0.405942,3.521614,4.851345>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
