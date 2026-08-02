#version 3.7;
global_settings { assumed_gamma 1.0 }
background { color rgbt <1,1,1,1> }
camera { orthographic location <8.255514,5.742966,-19.382512> look_at <0,0,0> right x*12.347377788770043 up y*9.260533341577533 }
light_source { <-7.178708,14.357416,-14.357416> color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }
light_source { <14.357416,-7.178708,-7.178708> color rgb <0.35,0.40,0.52> }
plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }
cylinder { <-2.406897,-1.740678,-2.201736>, <-3.212529,-1.891485,-1.496158>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.406897,-1.740678,-2.201736>, <-2.366183,-0.673159,-2.152398>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.406897,-1.740678,-2.201736>, <-2.691204,-2.020047,-3.196754>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.406897,-1.740678,-2.201736>, <-1.143154,-2.302642,-1.614342>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.143154,-2.302642,-1.614342>, <-1.321210,-3.187928,-0.738189>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.143154,-2.302642,-1.614342>, <0.162797,-1.758480,-1.625694>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.162797,-1.758480,-1.625694>, <0.637097,-2.259296,-0.875643>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.162797,-1.758480,-1.625694>, <0.849197,-0.515509,-2.020842>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.849197,-0.515509,-2.020842>, <0.226618,-0.112891,-2.796769>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.849197,-0.515509,-2.020842>, <2.251813,-0.872244,-2.596322>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.849197,-0.515509,-2.020842>, <1.051691,0.801597,-1.080866>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.251813,-0.872244,-2.596322>, <2.178180,-1.767825,-3.202709>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.251813,-0.872244,-2.596322>, <3.043450,-1.022925,-1.849590>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.251813,-0.872244,-2.596322>, <2.573941,-0.052533,-3.236012>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.051691,0.801597,-1.080866>, <0.851723,1.843384,-1.715089>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.051691,0.801597,-1.080866>, <1.565315,0.999642,0.206698>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.565315,0.999642,0.206698>, <1.569067,2.015851,0.366953>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.565315,0.999642,0.206698>, <1.978194,0.226326,1.359204>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.978194,0.226326,1.359204>, <1.703074,0.866357,2.218043>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.978194,0.226326,1.359204>, <3.494768,0.049576,1.319110>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.978194,0.226326,1.359204>, <1.256408,-1.086199,1.553064>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <3.494768,0.049576,1.319110>, <3.826623,-0.665445,2.063609>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <3.494768,0.049576,1.319110>, <3.769698,-0.301715,0.323298>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <3.494768,0.049576,1.319110>, <3.966179,1.016110,1.504850>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.256408,-1.086199,1.553064>, <1.933526,-2.146464,1.556262>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.256408,-1.086199,1.553064>, <-0.050201,-1.190603,1.112314>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.050201,-1.190603,1.112314>, <-0.264948,-2.181938,1.057237>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.050201,-1.190603,1.112314>, <-1.247216,-0.295113,1.198401>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.247216,-0.295113,1.198401>, <-1.969292,-1.046135,1.533895>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.247216,-0.295113,1.198401>, <-1.686615,0.353814,-0.160554>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.247216,-0.295113,1.198401>, <-1.384568,0.828423,2.242182>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.686615,0.353814,-0.160554>, <-1.306591,1.384272,-0.381451>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.686615,0.353814,-0.160554>, <-2.755470,0.423431,-0.325755>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.686615,0.353814,-0.160554>, <-1.204030,-0.139042,-0.917686>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.384568,0.828423,2.242182>, <-1.578392,0.722159,3.439145>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.384568,0.828423,2.242182>, <-1.618296,2.049033,1.587058>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.618296,2.049033,1.587058>, <-1.746520,2.028342,0.580483>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.618296,2.049033,1.587058>, <-2.149879,3.204888,2.158071>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.149879,3.204888,2.158071>, <-3.150658,2.978318,2.534009>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.149879,3.204888,2.158071>, <-1.422374,3.471869,2.913209>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.149879,3.204888,2.158071>, <-2.213131,3.966903,1.357464>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
sphere { <-2.406897,-1.740678,-2.201736>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-3.212529,-1.891485,-1.496158>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.366183,-0.673159,-2.152398>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.691204,-2.020047,-3.196754>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.143154,-2.302642,-1.614342>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.321210,-3.187928,-0.738189>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.162797,-1.758480,-1.625694>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.637097,-2.259296,-0.875643>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.849197,-0.515509,-2.020842>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.226618,-0.112891,-2.796769>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.251813,-0.872244,-2.596322>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.178180,-1.767825,-3.202709>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <3.043450,-1.022925,-1.849590>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.573941,-0.052533,-3.236012>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.051691,0.801597,-1.080866>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.851723,1.843384,-1.715089>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.565315,0.999642,0.206698>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.569067,2.015851,0.366953>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.978194,0.226326,1.359204>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.703074,0.866357,2.218043>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <3.494768,0.049576,1.319110>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <3.826623,-0.665445,2.063609>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <3.769698,-0.301715,0.323298>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <3.966179,1.016110,1.504850>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.256408,-1.086199,1.553064>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.933526,-2.146464,1.556262>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.050201,-1.190603,1.112314>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.264948,-2.181938,1.057237>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.247216,-0.295113,1.198401>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.969292,-1.046135,1.533895>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.686615,0.353814,-0.160554>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.306591,1.384272,-0.381451>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.755470,0.423431,-0.325755>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.204030,-0.139042,-0.917686>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.384568,0.828423,2.242182>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.578392,0.722159,3.439145>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.618296,2.049033,1.587058>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.746520,2.028342,0.580483>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.149879,3.204888,2.158071>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-3.150658,2.978318,2.534009>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.422374,3.471869,2.913209>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.213131,3.966903,1.357464>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
cylinder { <-2.131732,-1.542816,-1.953674>, <-1.903635,-1.378798,-1.748045>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.903635,-1.378798,-1.748045>, 0.12, <-1.759501,-1.275156,-1.618107>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-3.130541,-2.288758,-1.605024>, <-3.089400,-2.488110,-1.659653>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-3.089400,-2.488110,-1.659653>, 0.12, <-3.046454,-2.696206,-1.716678>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.260878,-0.511495,-2.525461>, <-2.207910,-0.430177,-2.713113>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.207910,-0.430177,-2.713113>, 0.12, <-2.152750,-0.345496,-2.908526>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.352501,-2.207388,-3.033709>, <-2.430494,-2.164249,-3.071253>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.430494,-2.164249,-3.071253>, 0.12, <-2.253078,-2.262380,-2.985849>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.242564,-2.195555,-1.220578>, <-1.298354,-2.135456,-0.999594>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.298354,-2.135456,-0.999594>, 0.12, <-1.350426,-2.079363,-0.793337>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.301278,-3.147139,-1.155728>, <-1.300441,-3.145425,-1.173270>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.300441,-3.145425,-1.173270>, 0.12, <-1.290000,-3.124059,-1.391981>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.027201,-2.141275,-1.518541>, <0.011921,-2.184412,-1.506467>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.011921,-2.184412,-1.506467>, 0.12, <-0.059105,-2.384924,-1.450339>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.957285,-2.194984,-1.139732>, <0.939507,-2.198555,-1.125069>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.939507,-2.198555,-1.125069>, 0.12, <1.107225,-2.164867,-1.263401>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.137484,-0.429301,-1.727826>, <1.097619,-0.441222,-1.768345>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.097619,-0.441222,-1.768345>, 0.12, <1.248627,-0.396065,-1.614861>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.220069,0.177392,-3.100237>, <0.220394,0.162992,-3.085183>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.220394,0.162992,-3.085183>, 0.12, <0.216963,0.315045,-3.244143>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.868507,-0.725737,-2.506806>, <1.935044,-0.751169,-2.522345>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.935044,-0.751169,-2.522345>, 0.12, <1.734265,-0.674427,-2.475456>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.374788,-2.000842,-3.491584>, <2.306704,-1.920150,-3.391549>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.306704,-1.920150,-3.391549>, 0.12, <2.409689,-2.042206,-3.542864>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.656005,-0.931910,-1.715416>, <2.669923,-0.935180,-1.720236>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.669923,-0.935180,-1.720236>, 0.12, <2.466976,-0.887505,-1.649955>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.841285,-0.307241,-3.436138>, <2.719305,-0.191026,-3.344827>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.719305,-0.191026,-3.344827>, 0.12, <2.859342,-0.324445,-3.449655>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.417827,0.659814,-1.230005>, <1.600811,0.588956,-1.304540>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.600811,0.588956,-1.304540>, 0.12, <1.792596,0.514689,-1.382660>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.434822,1.874776,-1.755192>, <0.450506,1.873595,-1.753683>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.450506,1.873595,-1.753683>, 0.12, <0.232129,1.890038,-1.774689>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.214811,1.018622,-0.023925>, <0.911197,1.035063,-0.223696>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.911197,1.035063,-0.223696>, 0.12, <0.727600,1.045005,-0.344498>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.537240,1.604097,0.290500>, <1.534350,1.566707,0.283558>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.534350,1.566707,0.283558>, 0.12, <1.517678,1.351026,0.243511>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.975759,0.463919,1.705533>, <1.973273,0.706508,2.059142>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.973273,0.706508,2.059142>, 0.12, <1.971998,0.830962,2.240552>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.029517,0.637138,2.086539>, <1.937159,0.701989,2.123745>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.937159,0.701989,2.123745>, 0.12, <2.108153,0.581922,2.054862>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <3.081893,-0.018514,1.283081>, <3.234087,0.006585,1.296362>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <3.234087,0.006585,1.296362>, 0.12, <3.017819,-0.029081,1.277489>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <3.481254,-0.782529,2.271963>, <3.640861,-0.728421,2.175675>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <3.640861,-0.728421,2.175675>, 0.12, <3.459953,-0.789750,2.284813>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <3.927304,-0.689376,0.359065>, <3.863846,-0.533290,0.344664>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <3.863846,-0.533290,0.344664>, 0.12, <3.946402,-0.736350,0.363399>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <4.194572,0.663841,1.492866>, <4.121372,0.776744,1.496707>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <4.121372,0.776744,1.496707>, 0.12, <4.241006,0.592222,1.490430>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.334276,-1.276519,1.186848>, <1.543779,-1.788572,0.201550>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.543779,-1.788572,0.201550>, 0.12, <1.584567,-1.888264,0.009723>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.800333,-1.876771,1.849392>, <1.705018,-1.683773,2.059161>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.705018,-1.683773,2.059161>, 0.12, <1.635250,-1.542505,2.212705>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.152055,-0.887815,1.384977>, <-0.268478,-0.541713,1.696644>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.268478,-0.541713,1.696644>, 0.12, <-0.321829,-0.383110,1.839467>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.670907,-2.111027,1.138290>, <-0.675275,-2.110264,1.139162>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.675275,-2.110264,1.139162>, 0.12, <-0.887920,-2.073121,1.181619>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.389513,-0.342321,0.806071>, <-1.391290,-0.342911,0.801171>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.391290,-0.342911,0.801171>, 0.12, <-1.465826,-0.367638,0.595665>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.277492,-0.825534,1.352932>, <-2.321217,-0.794236,1.327258>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.321217,-0.794236,1.327258>, 0.12, <-2.482655,-0.678683,1.232468>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.423808,0.026404,-0.148919>, <-1.413608,0.013696,-0.148467>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.413608,0.013696,-0.148467>, 0.12, <-1.275947,-0.157804,-0.142373>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.447450,1.067604,-0.144217>, <-1.507655,0.932256,-0.042820>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.507655,0.932256,-0.042820>, 0.12, <-1.581438,0.766382,0.081445>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.514962,0.255392,-0.025224>, <-2.527527,0.264172,-0.040926>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.527527,0.264172,-0.040926>, 0.12, <-2.401547,0.176151,0.116495>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.150766,-0.425154,-1.220511>, <-1.101748,-0.688463,-1.499203>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.101748,-0.688463,-1.499203>, 0.12, <-1.073848,-0.838330,-1.657825>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.741019,1.050303,2.252758>, <-1.841601,1.112911,2.255742>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.841601,1.112911,2.255742>, 0.12, <-2.028313,1.229134,2.261283>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.207827,0.535182,3.503344>, <-1.128744,0.495279,3.517045>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.128744,0.495279,3.517045>, 0.12, <-0.934639,0.397339,3.550674>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.219760,1.948805,1.500322>, <-1.127627,1.925635,1.480271>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.127627,1.925635,1.480271>, 0.12, <-0.918870,1.873135,1.434838>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.494249,2.322813,0.419096>, <-1.560302,2.245711,0.461352>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.560302,2.245711,0.461352>, 0.12, <-1.428160,2.399958,0.376817>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.793790,3.010447,2.266669>, <-1.891271,3.063676,2.236940>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.891271,3.063676,2.236940>, 0.12, <-1.704748,2.961826,2.293824>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-3.181340,3.294758,2.808462>, <-3.186936,3.352471,2.858518>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-3.186936,3.352471,2.858518>, 0.12, <-3.203008,3.518225,3.002280>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.568952,3.763160,3.177906>, <-1.661343,3.946768,3.344751>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.661343,3.946768,3.344751>, 0.12, <-1.738122,4.099349,3.483402>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.308171,4.143618,1.726434>, <-2.319591,4.164851,1.770767>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.319591,4.164851,1.770767>, 0.12, <-2.369374,4.257416,1.964038>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
