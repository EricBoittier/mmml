#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White}
camera {orthographic
  right -2.05*x up 0.32*y
  direction 1.00*z
  location <0,0,60.00> look_at <0,0,0>}


light_source {<  2.00,   3.00,  40.00> color White
  area_light <0.70, 0, 0>, <0, 0.70, 0>, 3, 3
  adaptive 1 jitter}
// no fog
#declare simple = finish {phong 0.7 ambient 0.4 diffuse 0.55}
#declare pale = finish {ambient 0.9 diffuse 0.30 roughness 0.001 specular 0.2 }
#declare intermediate = finish {ambient 0.4 diffuse 0.6 specular 0.1 roughness 0.04}
#declare vmd = finish {ambient 0.2 diffuse 0.80 phong 0.25 phong_size 10.0 specular 0.2 roughness 0.1}
#declare jmol = finish {ambient 0.4 diffuse 0.6 specular 1 roughness 0.001 metallic}
#declare ase2 = finish {ambient 0.2 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient 0.4 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}
#declare glass = finish {ambient 0.4 diffuse 0.35 specular 1.0 roughness 0.001}
#declare glass2 = finish {ambient 0.3 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}
#declare Rcell = 0.000;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     translate LOC}
#end

// no cell vertices
atom(<  0.00,   0.00,  -0.83>, 0.15, rgb <0.56, 0.56, 0.56>, 0.0, jmol) // #0
atom(<  0.83,   0.00,   0.00>, 0.15, rgb <1.00, 0.05, 0.05>, 0.0, jmol) // #1
atom(< -0.83,   0.00,  -1.65>, 0.15, rgb <1.00, 0.05, 0.05>, 0.0, jmol) // #2
cylinder {<  0.00,   0.00,  -0.83>, <  0.41,   0.00,  -0.41>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{jmol}}}
cylinder {<  0.83,   0.00,   0.00>, <  0.41,   0.00,  -0.41>, Rbond texture{pigment {color rgb <1.00, 0.05, 0.05> transmit 0.0} finish{jmol}}}
cylinder {<  0.00,   0.00,  -0.83>, < -0.41,   0.00,  -1.24>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{jmol}}}
cylinder {< -0.83,   0.00,  -1.65>, < -0.41,   0.00,  -1.24>, Rbond texture{pigment {color rgb <1.00, 0.05, 0.05> transmit 0.0} finish{jmol}}}
// no constraints


// Enhanced rendering settings
// Added by ase_povray_viz.py


// Soft area lighting for better appearance
#declare light_color = rgb <1.0, 1.0, 1.0>;

// Main key light (soft area light)
light_source {
    <20, 30, 40>
    color light_color * 0.8
    area_light <8, 0, 0>, <0, 8, 0>, 5, 5
    adaptive 1
    jitter
    fade_distance 60
    fade_power 1
}

// Fill light (softer, from opposite side)
light_source {
    <-15, 20, -30>
    color light_color * 0.4
    area_light <6, 0, 0>, <0, 6, 0>, 4, 4
    adaptive 1
    jitter
    fade_distance 50
    fade_power 1
}

// Back rim light (subtle)
light_source {
    <-10, -15, -40>
    color light_color * 0.3
    area_light <5, 0, 0>, <0, 5, 0>, 3, 3
    adaptive 1
    jitter
}

// Ambient illumination
global_settings {
    ambient_light rgb <0.15, 0.15, 0.15>
    max_trace_level 15
}

// Enhanced charge texture with anisotropic properties
#declare charge_finish = finish {
    ambient 0.2
    diffuse 0.65
    specular 0.5
    roughness 0.005
    metallic 0.15
    phong 0.6
    phong_size 80
    brilliance 1.2
}

// Enhanced ESP surface finish
#declare esp_finish = finish {
    ambient 0.25
    diffuse 0.6
    specular 0.3
    roughness 0.02
    brilliance 1.0
}

sphere {
  <0.0000, 2.3800, 0.0000>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1270, 2.3752, -0.0808>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0901, 2.3705, -0.1926>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0335, 2.3657, -0.2582>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1925, 2.3610, -0.2307>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3199, 2.3562, -0.1021>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3555, 2.3514, 0.0937>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2695, 2.3467, 0.2913>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0759, 2.3419, 0.4172>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1695, 2.3372, 0.4164>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3862, 2.3324, 0.2743>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4959, 2.3276, 0.0254>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4510, 2.3229, -0.2555>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2532, 2.3181, -0.4761>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0435, 2.3133, -0.5576>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3476, 2.3086, -0.4626>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5590, 2.3038, -0.2104>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6023, 2.2991, 0.1263>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4530, 2.2943, 0.4420>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1488, 2.2895, 0.6327>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2194, 2.2848, 0.6293>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5356, 2.2800, 0.4232>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6946, 2.2753, 0.0712>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6380, 2.2705, -0.3196>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3745, 2.2657, -0.6250>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0198, 2.2610, -0.7430>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4235, 2.2562, -0.6281>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7073, 2.2515, -0.3083>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7759, 2.2467, 0.1217>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5995, 2.2419, 0.5280>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2261, 2.2372, 0.7800>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2315, 2.2324, 0.7919>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6300, 2.2276, 0.5523>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8404, 2.2229, 0.1298>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7901, 2.2181, -0.3465>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4874, 2.2134, -0.7265>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0216, 2.2086, -0.8866>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4637, 2.2038, -0.7697>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8147, 2.1991, -0.4058>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9166, 2.1943, 0.0961>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7306, 2.1896, 0.5800>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3087, 2.1848, 0.8920>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2208, 2.1800, 0.9290>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6927, 2.1753, 0.6729>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9560, 2.1705, 0.1981>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9228, 2.1658, -0.3497>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5972, 2.1610, -0.7987>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0760, 2.1562, -1.0046>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4799, 2.1515, -0.8974>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8953, 2.1467, -0.5045>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0360, 2.1420, 0.0554>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8524, 2.1372, 0.6085>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3963, 2.1324, 0.9799>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1933, 2.1277, 1.0488>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7325, 2.1229, 0.7881>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0501, 2.1181, 0.2742>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0420, 2.1134, -0.3351>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7052, 2.1086, -0.8490>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1405, 2.1039, -1.1038>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4778, 2.0991, -1.0148>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9553, 2.0943, -0.6046>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1393, 2.0896, 0.0026>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9672, 2.0848, 0.6184>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4878, 2.0801, 1.0487>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1524, 2.0753, 1.1551>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7540, 2.0705, 0.8993>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1268, 2.0658, 0.3566>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1502, 2.0610, -0.3060>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8118, 2.0563, -0.8816>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2132, 2.0515, -1.1876>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4606, 2.0467, -1.1239>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9983, 2.0420, -0.7057>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2291, 2.0372, -0.0600>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0760, 2.0325, 0.6129>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5826, 2.0277, 1.1016>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1003, 2.0229, 1.2498>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7601, 2.0182, 1.0068>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1888, 2.0134, 0.4442>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2488, 2.0086, -0.2648>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9169, 2.0039, -0.8990>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2928, 1.9991, -1.2578>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4305, 1.9944, -1.2254>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0266, 1.9896, -0.8074>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3068, 1.9848, -0.1309>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1793, 1.9801, 0.5941>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6798, 1.9753, 1.1404>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0387, 1.9706, 1.3341>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7526, 1.9658, 1.1107>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2375, 1.9610, 0.5359>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3386, 1.9563, -0.2131>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0202, 1.9515, -0.9028>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3781, 1.9468, -1.3159>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3891, 1.9420, -1.3197>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0417, 1.9372, -0.9091>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3734, 1.9325, -0.2089>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2770, 1.9277, 0.5635>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7786, 1.9229, 1.1664>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0311, 1.9182, 1.4085>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7331, 1.9134, 1.2107>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2741, 1.9087, 0.6309>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4200, 1.9039, -0.1522>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1214, 1.8991, -0.8945>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4681, 1.8944, -1.3626>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3378, 1.8896, -1.4070>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0447, 1.8849, -1.0101>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4296, 1.8801, -0.2928>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3692, 1.8753, 0.5224>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8785, 1.8706, 1.1805>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1080, 1.8658, 1.4736>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7026, 1.8611, 1.3066>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2993, 1.8563, 0.7283>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4931, 1.8515, -0.0832>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2199, 1.8468, -0.8750>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5618, 1.8420, -1.3985>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2776, 1.8373, -1.4872>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0364, 1.8325, -1.1100>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4758, 1.8277, -0.3818>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4556, 1.8230, 0.4718>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9785, 1.8182, 1.1836>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1910, 1.8134, 1.5295>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6621, 1.8087, 1.3981>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3138, 1.8039, 0.8272>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5580, 1.7992, -0.0072>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3153, 1.7944, -0.8452>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6584, 1.7896, -1.4241>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2095, 1.7849, -1.5604>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0177, 1.7801, -1.2082>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5123, 1.7754, -0.4748>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5360, 1.7706, 0.4125>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0782, 1.7658, 1.1763>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2793, 1.7611, 1.5764>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6124, 1.7563, 1.4848>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3179, 1.7516, 0.9271>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6148, 1.7468, 0.0751>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4072, 1.7420, -0.8059>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7572, 1.7373, -1.4398>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1343, 1.7325, -1.6263>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9892, 1.7277, -1.3041>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5393, 1.7230, -0.5712>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6102, 1.7182, 0.3453>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1768, 1.7135, 1.1591>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3719, 1.7087, 1.6144>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5543, 1.7039, 1.5664>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3123, 1.6992, 1.0272>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6634, 1.6944, 0.1628>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4952, 1.6897, -0.7576>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8574, 1.6849, -1.4458>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0528, 1.6801, -1.6849>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9515, 1.6754, -1.3972>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5571, 1.6706, -0.6701>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6780, 1.6659, 0.2711>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2738, 1.6611, 1.1325>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4682, 1.6563, 1.6437>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4884, 1.6516, 1.6426>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2972, 1.6468, 1.1268>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7038, 1.6421, 0.2552>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5787, 1.6373, -0.7010>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9583, 1.6325, -1.4426>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0344, 1.6278, -1.7360>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9051, 1.6230, -1.4870>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5657, 1.6182, -0.7708>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7392, 1.6135, 0.1905>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3686, 1.6087, 1.0969>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5675, 1.6040, 1.6642>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4155, 1.5992, 1.7130>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2731, 1.5944, 1.2253>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7360, 1.5897, 0.3514>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6574, 1.5849, -0.6368>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0592, 1.5802, -1.4303>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1263, 1.5754, -1.7795>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8505, 1.5706, -1.5729>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5655, 1.5659, -0.8727>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7935, 1.5611, 0.1041>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4607, 1.5564, 1.0529>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6689, 1.5516, 1.6762>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3360, 1.5468, 1.7773>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2402, 1.5421, 1.3222>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7600, 1.5373, 0.4510>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7309, 1.5326, -0.5654>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1595, 1.5278, -1.4092>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2225, 1.5230, -1.8153>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7883, 1.5183, -1.6546>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5566, 1.5135, -0.9751>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8406, 1.5087, 0.0127>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5495, 1.5040, 1.0008>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7719, 1.4992, 1.6796>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2507, 1.4945, 1.8352>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1991, 1.4897, 1.4168>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7758, 1.4849, 0.5530>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7988, 1.4802, -0.4875>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2587, 1.4754, -1.3796>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3222, 1.4707, -1.8433>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7189, 1.4659, -1.7317>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5391, 1.4611, -1.0774>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8805, 1.4564, -0.0831>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6345, 1.4516, 0.9410>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8758, 1.4469, 1.6745>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1602, 1.4421, 1.8866>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1499, 1.4373, 1.5087>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7834, 1.4326, 0.6570>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8609, 1.4278, -0.4036>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3561, 1.4230, -1.3418>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4248, 1.4183, -1.8634>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6429, 1.4135, -1.8036>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5133, 1.4088, -1.1789>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9130, 1.4040, -0.1828>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7153, 1.3992, 0.8741>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9799, 1.3945, 1.6612>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0649, 1.3897, 1.9310>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0931, 1.3850, 1.5973>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7828, 1.3802, 0.7623>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9167, 1.3754, -0.3142>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4511, 1.3707, -1.2961>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5297, 1.3659, -1.8757>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5608, 1.3612, -1.8701>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4794, 1.3564, -1.2791>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9380, 1.3516, -0.2857>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7915, 1.3469, 0.8005>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0838, 1.3421, 1.6397>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0344, 1.3374, 1.9684>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0292, 1.3326, 1.6821>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7741, 1.3278, 0.8683>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9661, 1.3231, -0.2199>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5433, 1.3183, -1.2428>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6362, 1.3135, -1.8800>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4730, 1.3088, -1.9307>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4376, 1.3040, -1.3774>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9553, 1.2993, -0.3911>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8626, 1.2945, 0.7206>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1867, 1.2897, 1.6102>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1373, 1.2850, 1.9986>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9583, 1.2802, 1.7627>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7573, 1.2755, 0.9744>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0087, 1.2707, -0.1213>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6322, 1.2659, -1.1822>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7438, 1.2612, -1.8763>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3801, 1.2564, -1.9853>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3883, 1.2517, -1.4732>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9649, 1.2469, -0.4986>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9283, 1.2421, 0.6348>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2882, 1.2374, 1.5729>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2430, 1.2326, 2.0214>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8811, 1.2278, 1.8386>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7327, 1.2231, 1.0799>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0444, 1.2183, -0.0188>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7173, 1.2136, -1.1147>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8518, 1.2088, -1.8648>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2826, 1.2040, -2.0334>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3316, 1.1993, -1.5662>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9668, 1.1945, -0.6076>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9883, 1.1898, 0.5437>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3876, 1.1850, 1.5280>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3512, 1.1802, 2.0367>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7979, 1.1755, 1.9095>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7003, 1.1707, 1.1844>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0730, 1.1660, 0.0869>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7980, 1.1612, -1.0407>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9598, 1.1564, -1.8455>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1810, 1.1517, -2.0749>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2680, 1.1469, -1.6557>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9609, 1.1422, -0.7173>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0421, 1.1374, 0.4478>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4845, 1.1326, 1.4757>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4611, 1.1279, 2.0444>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7091, 1.1231, 1.9749>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6603, 1.1183, 1.2873>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0943, 1.1136, 0.1953>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8741, 1.1088, -0.9605>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0671, 1.1041, -1.8185>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0758, 1.0993, -2.1095>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1976, 1.0945, -1.7413>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9473, 1.0898, -0.8274>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0896, 1.0850, 0.3475>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5784, 1.0803, 1.4163>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5723, 1.0755, 2.0446>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6152, 1.0707, 2.0346>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6129, 1.0660, 1.3880>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1082, 1.0612, 0.3058>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9451, 1.0565, -0.8745>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1731, 1.0517, -1.7838>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0324, 1.0469, -2.1371>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1209, 1.0422, -1.8226>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9261, 1.0374, -0.9372>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1305, 1.0327, 0.2433>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6688, 1.0279, 1.3502>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6841, 1.0231, 2.0371>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5167, 1.0184, 2.0882>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5583, 1.0136, 1.4861>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1147, 1.0088, 0.4180>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0107, 1.0041, -0.7832>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2775, 0.9993, -1.7418>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1432, 0.9946, -2.1575>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0382, 0.9898, -1.8992>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8972, 0.9850, -1.0462>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1645, 0.9803, 0.1357>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7552, 0.9755, 1.2775>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7961, 0.9708, 2.0219>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4139, 0.9660, 2.1354>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4968, 0.9612, 1.5811>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1136, 0.9565, 0.5312>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0705, 0.9517, -0.6869>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3796, 0.9470, -1.6925>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2559, 0.9422, -2.1705>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9500, 0.9374, -1.9706>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8609, 0.9327, -1.1539>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1915, 0.9279, 0.0252>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8372, 0.9231, 1.1986>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9077, 0.9184, 1.9993>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3075, 0.9136, 2.1760>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4287, 0.9089, 1.6725>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1050, 0.9041, 0.6449>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1241, 0.8993, -0.5862>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4789, 0.8946, -1.6361>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3701, 0.8898, -2.1762>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8565, 0.8851, -2.0365>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8173, 0.8803, -1.2597>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2114, 0.8755, -0.0875>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9145, 0.8708, 1.1140>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0183, 0.8660, 1.9691>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1979, 0.8613, 2.2099>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3542, 0.8565, 1.7598>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0889, 0.8517, 0.7587>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1715, 0.8470, -0.4814>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5751, 0.8422, -1.5729>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4851, 0.8375, -2.1743>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7583, 0.8327, -2.0967>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7666, 0.8279, -1.3632>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2239, 0.8232, -0.2022>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9865, 0.8184, 1.0239>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1275, 0.8136, 1.9316>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0855, 0.8089, 2.2367>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2736, 0.8041, 1.8427>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0652, 0.7994, 0.8719>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2122, 0.7946, -0.3731>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6676, 0.7898, -1.5033>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6006, 0.7851, -2.1650>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6557, 0.7803, -2.1507>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7089, 0.7756, -1.4638>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2291, 0.7708, -0.3181>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0531, 0.7660, 0.9287>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2347, 0.7613, 1.8869>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0291, 0.7565, 2.2564>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1874, 0.7518, 1.9208>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0342, 0.7470, 0.9841>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2461, 0.7422, -0.2617>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7559, 0.7375, -1.4273>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7160, 0.7327, -2.1482>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5493, 0.7279, -2.1984>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6445, 0.7232, -1.5611>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2269, 0.7184, -0.4348>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1138, 0.7137, 0.8289>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3395, 0.7089, 1.8351>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1453, 0.7041, 2.2688>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0958, 0.6994, 1.9936>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9959, 0.6946, 1.0946>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2730, 0.6899, -0.1477>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8397, 0.6851, -1.3455>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8306, 0.6803, -2.1240>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4394, 0.6756, -2.2394>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5737, 0.6708, -1.6547>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2173, 0.6661, -0.5518>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1683, 0.6613, 0.7249>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4413, 0.6565, 1.7765>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2628, 0.6518, 2.2739>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9993, 0.6470, 2.0608>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9505, 0.6423, 1.2032>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2928, 0.6375, -0.0317>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9187, 0.6327, -1.2581>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9442, 0.6280, -2.0925>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3266, 0.6232, -2.2736>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4967, 0.6184, -1.7440>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2002, 0.6137, -0.6686>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2165, 0.6089, 0.6171>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5398, 0.6042, 1.7113>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3809, 0.5994, 2.2716>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8983, 0.5946, 2.1222>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8980, 0.5899, 1.3092>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3054, 0.5851, 0.0860>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9923, 0.5804, -1.1654>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0561, 0.5756, -2.0537>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2113, 0.5708, -2.3008>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4140, 0.5661, -1.8288>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1757, 0.5613, -0.7846>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2580, 0.5566, 0.5060>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6344, 0.5518, 1.6397>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4992, 0.5470, 2.2618>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7932, 0.5423, 2.1774>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8388, 0.5375, 1.4122>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3106, 0.5328, 0.2047>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0604, 0.5280, -1.0679>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1658, 0.5232, -2.0079>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0941, 0.5185, -2.3209>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3257, 0.5137, -1.9087>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1440, 0.5089, -0.8993>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2927, 0.5042, 0.3921>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7248, 0.4994, 1.5621>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6172, 0.4947, 2.2447>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6844, 0.4899, 2.2262>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7731, 0.4851, 1.5117>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3084, 0.4804, 0.3240>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1225, 0.4756, -0.9660>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2729, 0.4709, -1.9551>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0247, 0.4661, -2.3338>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2323, 0.4613, -1.9832>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1050, 0.4566, -1.0123>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3204, 0.4518, 0.2758>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8105, 0.4471, 1.4787>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7343, 0.4423, 2.2203>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5724, 0.4375, 2.2683>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7011, 0.4328, 1.6073>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2988, 0.4280, 0.4434>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1784, 0.4232, -0.8601>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3769, 0.4185, -1.8956>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1444, 0.4137, -2.3393>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1341, 0.4090, -2.0521>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0590, 0.4042, -1.1231>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3409, 0.3994, 0.1576>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8913, 0.3947, 1.3899>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8501, 0.3899, 2.1886>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4576, 0.3852, 2.3036>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6230, 0.3804, 1.6987>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2819, 0.3756, 0.5623>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2279, 0.3709, -0.7505>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4774, 0.3661, -1.8296>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2647, 0.3614, -2.3375>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0316, 0.3566, -2.1150>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0062, 0.3518, -1.2312>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3543, 0.3471, 0.0380>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9666, 0.3423, 1.2960>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9640, 0.3376, 2.1497>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3405, 0.3328, 2.3319>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5393, 0.3280, 1.7853>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2577, 0.3233, 0.6804>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2707, 0.3185, -0.6379>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5740, 0.3137, -1.7574>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3849, 0.3090, -2.3283>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9251, 0.3042, -2.1716>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9467, 0.2995, -1.3361>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3602, 0.2947, -0.0826>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0363, 0.2899, 1.1974>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0757, 0.2852, 2.1038>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2216, 0.2804, 2.3530>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4503, 0.2757, 1.8669>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2262, 0.2709, 0.7970>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3066, 0.2661, -0.5225>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6662, 0.2614, -1.6793>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5046, 0.2566, -2.3117>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8152, 0.2519, -2.2218>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8807, 0.2471, -1.4375>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3589, 0.2423, -0.2035>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1000, 0.2376, 1.0946>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1846, 0.2328, 2.0511>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1014, 0.2280, 2.3669>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3562, 0.2233, 1.9430>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1876, 0.2185, 0.9117>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3355, 0.2138, -0.4050>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7536, 0.2090, -1.5955>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6233, 0.2042, -2.2878>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7021, 0.1995, -2.2653>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8086, 0.1947, -1.5348>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3501, 0.1900, -0.3243>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1574, 0.1852, 0.9879>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2902, 0.1804, 1.9918>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0196, 0.1757, 2.3734>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2575, 0.1709, 2.0134>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1420, 0.1662, 1.0240>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3573, 0.1614, -0.2857>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8359, 0.1566, -1.5065>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7406, 0.1519, -2.2567>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5865, 0.1471, -2.3019>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7305, 0.1424, -1.6277>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3341, 0.1376, -0.4446>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2082, 0.1328, 0.8777>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3923, 0.1281, 1.9260>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1410, 0.1233, 2.3726>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1546, 0.1185, 2.0778>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0896, 0.1138, 1.1336>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3718, 0.1090, -0.1651>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9127, 0.1043, -1.4124>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8558, 0.0995, -2.2186>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4687, 0.0947, -2.3315>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6469, 0.0900, -1.7158>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3107, 0.0852, -0.5638>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2524, 0.0805, 0.7646>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4902, 0.0757, 1.8542>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2622, 0.0709, 2.3644>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0479, 0.0662, 2.1359>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0307, 0.0614, 1.2398>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3789, 0.0567, -0.0438>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9838, 0.0519, -1.3139>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9687, 0.0471, -2.1734>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3492, 0.0424, -2.3539>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5581, 0.0376, -1.7987>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2801, 0.0329, -0.6814>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2897, 0.0281, 0.6489>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5837, 0.0233, 1.7765>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3828, 0.0186, 2.3489>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9378, 0.0138, 2.1874>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9653, 0.0090, 1.3423>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3787, 0.0043, 0.0777>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0488, -0.0005, -1.2112>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0786, -0.0052, -2.1216>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2286, -0.0100, -2.3690>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4644, -0.0148, -1.8761>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2425, -0.0195, -0.7970>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3199, -0.0243, 0.5311>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6723, -0.0290, 1.6932>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5022, -0.0338, 2.3262>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8248, -0.0386, 2.2322>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8939, -0.0433, 1.4407>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3712, -0.0481, 0.1990>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1074, -0.0528, -1.1047>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1852, -0.0576, -2.0631>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1073, -0.0624, -2.3768>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3661, -0.0671, -1.9477>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1980, -0.0719, -0.9101>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3429, -0.0767, 0.4118>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7557, -0.0814, 1.6048>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6200, -0.0862, 2.2962>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7093, -0.0909, 2.2700>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8167, -0.0957, 1.5345>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3563, -0.1005, 0.3196>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1595, -0.1052, -0.9949>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2880, -0.1100, -1.9983>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0142, -0.1147, -2.3772>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2638, -0.1195, -2.0132>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1467, -0.1243, -1.0202>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3586, -0.1290, 0.2913>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8335, -0.1338, 1.5116>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7356, -0.1385, 2.2592>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5918, -0.1433, 2.3008>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7340, -0.1481, 1.6235>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3342, -0.1528, 0.4390>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2048, -0.1576, -0.8823>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3866, -0.1623, -1.9275>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1354, -0.1671, -2.3703>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1578, -0.1719, -2.0723>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0888, -0.1766, -1.1269>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3670, -0.1814, 0.1703>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9054, -0.1862, 1.4139>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8487, -0.1909, 2.2153>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4728, -0.1957, 2.3243>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6462, -0.2004, 1.7071>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3049, -0.2052, 0.5567>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2431, -0.2100, -0.7672>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4806, -0.2147, -1.8510>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2558, -0.2195, -2.3560>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0485, -0.2242, -2.1248>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0247, -0.2290, -1.2298>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3680, -0.2338, 0.0491>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9713, -0.2385, 1.3121>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9588, -0.2433, 2.1647>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3528, -0.2480, 2.3406>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5536, -0.2528, 1.7852>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2685, -0.2576, 0.6722>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2744, -0.2623, -0.6502>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5697, -0.2671, -1.7690>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3749, -0.2719, -2.3345>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9365, -0.2766, -2.1705>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9546, -0.2814, -1.3284>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3616, -0.2861, -0.0717>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0306, -0.2909, 1.2068>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0654, -0.2957, 2.1076>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2322, -0.3004, 2.3495>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4566, -0.3052, 1.8573>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2253, -0.3099, 0.7851>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2984, -0.3147, -0.5318>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6534, -0.3195, -1.6819>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4922, -0.3242, -2.3059>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8221, -0.3290, -2.2092>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8788, -0.3337, -1.4224>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3480, -0.3385, -0.1917>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0834, -0.3433, 1.0982>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1681, -0.3480, 2.0442>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1116, -0.3528, 2.3511>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3556, -0.3575, 1.9232>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1754, -0.3623, 0.8949>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3151, -0.3671, -0.4124>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7315, -0.3718, -1.5900>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6073, -0.3766, -2.2702>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7058, -0.3814, -2.2407>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7976, -0.3861, -1.5113>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.3271, -0.3909, -0.3102>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1293, -0.3956, 0.9869>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2664, -0.4004, 1.9749>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0086, -0.4052, 2.3452>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2511, -0.4099, 1.9827>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1190, -0.4147, 1.0011>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3244, -0.4194, -0.2926>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8035, -0.4242, -1.4939>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7197, -0.4290, -2.2277>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5882, -0.4337, -2.2650>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7113, -0.4385, -1.5948>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2991, -0.4432, -0.4270>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1682, -0.4480, 0.8734>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3601, -0.4528, 1.8999>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1278, -0.4575, 2.3321>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1434, -0.4623, 2.0355>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0564, -0.4670, 1.1033>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3264, -0.4718, -0.1727>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8694, -0.4766, -1.3938>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8289, -0.4813, -2.1785>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4697, -0.4861, -2.2820>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6204, -0.4909, -1.6726>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2640, -0.4956, -0.5413>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1999, -0.5004, 0.7581>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4486, -0.5051, 1.8195>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2456, -0.5099, 2.3117>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0331, -0.5147, 2.0814>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9879, -0.5194, 1.2012>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3209, -0.5242, -0.0535>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9287, -0.5289, -1.2902>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9344, -0.5337, -2.1228>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3508, -0.5385, -2.2916>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5252, -0.5432, -1.7444>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2221, -0.5480, -0.6529>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2243, -0.5527, 0.6414>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5317, -0.5575, 1.7342>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3614, -0.5623, 2.2842>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9206, -0.5670, 2.1202>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9138, -0.5718, 1.2942>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.3082, -0.5766, 0.0648>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9813, -0.5813, -1.1836>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0359, -0.5861, -2.0610>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2320, -0.5908, -2.2938>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4262, -0.5956, -1.8099>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1736, -0.6004, -0.7612>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2413, -0.6051, 0.5240>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6090, -0.6099, 1.6443>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4748, -0.6146, 2.2497>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8064, -0.6194, 2.1519>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8343, -0.6242, 1.3820>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2882, -0.6289, 0.1815>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0269, -0.6337, -1.0744>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1329, -0.6384, -1.9933>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1139, -0.6432, -2.2886>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3237, -0.6480, -1.8688>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1187, -0.6527, -0.8658>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2510, -0.6575, 0.4063>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6802, -0.6622, 1.5501>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5853, -0.6670, 2.2084>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6910, -0.6718, 2.1762>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7500, -0.6765, 1.4644>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2611, -0.6813, 0.2961>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0655, -0.6861, -0.9631>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2250, -0.6908, -1.9200>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0031, -0.6956, -2.2761>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2182, -0.7003, -1.9209>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0576, -0.7051, -0.9663>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2532, -0.7099, 0.2889>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7450, -0.7146, 1.4522>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6925, -0.7194, 2.1604>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5749, -0.7241, 2.1931>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6610, -0.7289, 1.5408>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.2270, -0.7337, 0.4083>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0967, -0.7384, -0.8502>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3119, -0.7432, -1.8415>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1184, -0.7479, -2.2563>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1103, -0.7527, -1.9660>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9906, -0.7575, -1.0621>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2481, -0.7622, 0.1721>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8031, -0.7670, 1.3509>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7958, -0.7718, 2.1061>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4586, -0.7765, 2.2025>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5680, -0.7813, 1.6111>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1861, -0.7860, 0.5174>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1206, -0.7908, -0.7362>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3931, -0.7956, -1.7581>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2316, -0.8003, -2.2294>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0004, -0.8051, -2.0039>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9181, -0.8098, -1.1530>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2355, -0.8146, 0.0567>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8544, -0.8194, 1.2467>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8948, -0.8241, 2.0456>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3427, -0.8289, 2.2045>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4712, -0.8336, 1.6748>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1385, -0.8384, 0.6230>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1371, -0.8432, -0.6216>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4683, -0.8479, -1.6702>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3422, -0.8527, -2.1955>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8889, -0.8574, -2.0345>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8405, -0.8622, -1.2384>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.2157, -0.8670, -0.0571>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8986, -0.8717, 1.1401>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9891, -0.8765, 1.9793>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2276, -0.8813, 2.1991>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3711, -0.8860, 1.7319>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0846, -0.8908, 0.7247>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1460, -0.8955, -0.5070>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5373, -0.9003, -1.5782>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4496, -0.9051, -2.1548>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7765, -0.9098, -2.0576>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7580, -0.9146, -1.3182>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1888, -0.9193, -0.1685>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9355, -0.9241, 1.0317>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0784, -0.9289, 1.9075>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1139, -0.9336, 2.1863>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2683, -0.9384, 1.7819>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0247, -0.9431, 0.8220>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1475, -0.9479, -0.3928>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5996, -0.9527, -1.4826>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5534, -0.9574, -2.1075>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6636, -0.9622, -2.0732>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6710, -0.9669, -1.3918>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1548, -0.9717, -0.2772>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9650, -0.9765, 0.9218>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1621, -0.9812, 1.8306>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0020, -0.9860, 2.1662>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1632, -0.9908, 1.8248>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9589, -0.9955, 0.9144>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1414, -1.0003, -0.2796>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6551, -1.0050, -1.3838>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6531, -1.0098, -2.0538>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5507, -1.0146, -2.0813>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5801, -1.0193, -1.4590>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1141, -1.0241, -0.3825>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9870, -1.0288, 0.8111>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2399, -1.0336, 1.7489>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1073, -1.0384, 2.1389>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0563, -1.0431, 1.8603>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8876, -1.0479, 1.0016>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1279, -1.0526, -0.1679>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7035, -1.0574, -1.2824>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7482, -1.0622, -1.9941>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4385, -1.0669, -2.0818>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4856, -1.0717, -1.5195>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0667, -1.0765, -0.4841>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0013, -1.0812, 0.7000>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3114, -1.0860, 1.6629>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2138, -1.0907, 2.1045>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9481, -1.0955, 1.8882>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8113, -1.1003, 1.0831>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1071, -1.1050, -0.0583>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7446, -1.1098, -1.1787>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8383, -1.1145, -1.9286>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3274, -1.1193, -2.0747>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3881, -1.1241, -1.5729>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0130, -1.1288, -0.5814>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0081, -1.1336, 0.5891>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3764, -1.1383, 1.5729>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3167, -1.1431, 2.0633>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8392, -1.1479, 1.9086>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7302, -1.1526, 1.1585>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0791, -1.1574, 0.0488>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7781, -1.1621, -1.0734>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9230, -1.1669, -1.8576>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2180, -1.1717, -2.0601>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2880, -1.1764, -1.6191>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9532, -1.1812, -0.6740>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0071, -1.1860, 0.4790>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4345, -1.1907, 1.4795>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4157, -1.1955, 2.0155>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7301, -1.2002, 1.9212>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6449, -1.2050, 1.2275>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0439, -1.2098, 0.1526>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8040, -1.2145, -0.9669>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0018, -1.2193, -1.7816>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1108, -1.2240, -2.0381>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1858, -1.2288, -1.6578>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8876, -1.2336, -0.7613>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9985, -1.2383, 0.3703>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4854, -1.2431, 1.3831>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5102, -1.2478, 1.9614>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6213, -1.2526, 1.9260>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5557, -1.2574, 1.2897>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0019, -1.2621, 0.2528>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8221, -1.2669, -0.8599>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0743, -1.2717, -1.7010>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0065, -1.2764, -2.0088>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0820, -1.2812, -1.6889>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8167, -1.2859, -0.8429>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9822, -1.2907, 0.2634>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5288, -1.2955, 1.2842>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5997, -1.3002, 1.9011>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5136, -1.3050, 1.9229>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4631, -1.3097, 1.3447>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9532, -1.3145, 0.3488>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8323, -1.3193, -0.7528>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1401, -1.3240, -1.6160>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0944, -1.3288, -1.9723>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9773, -1.3335, -1.7120>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7406, -1.3383, -0.9185>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9584, -1.3431, 0.1590>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5644, -1.3478, 1.1833>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6837, -1.3526, 1.8351>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4073, -1.3573, 1.9121>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3676, -1.3621, 1.3923>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8980, -1.3669, 0.4400>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8345, -1.3716, -0.6464>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1988, -1.3764, -1.5273>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1914, -1.3812, -1.9288>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8722, -1.3859, -1.7271>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6600, -1.3907, -0.9874>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9271, -1.3954, 0.0577>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5922, -1.4002, 1.0811>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7618, -1.4050, 1.7636>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3032, -1.4097, 1.8934>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2698, -1.4145, 1.4322>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8367, -1.4192, 0.5259>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8286, -1.4240, -0.5411>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2502, -1.4288, -1.4353>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2839, -1.4335, -1.8785>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7672, -1.4383, -1.7341>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5751, -1.4430, -1.0494>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8886, -1.4478, -0.0399>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6118, -1.4526, 0.9780>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8334, -1.4573, 1.6870>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2019, -1.4621, 1.8671>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1701, -1.4668, 1.4641>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7696, -1.4716, 0.6061>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8147, -1.4764, -0.4375>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2937, -1.4811, -1.3405>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3714, -1.4859, -1.8217>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6631, -1.4907, -1.7328>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4864, -1.4954, -1.1039>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8428, -1.5002, -0.1333>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6231, -1.5049, 0.8748>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8981, -1.5097, 1.6058>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1039, -1.5145, 1.8330>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0692, -1.5192, 1.4877>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6970, -1.5240, 0.6799>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7928, -1.5287, -0.3364>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3292, -1.5335, -1.2434>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4531, -1.5383, -1.7586>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5603, -1.5430, -1.7232>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3945, -1.5478, -1.1507>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7902, -1.5525, -0.2219>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6259, -1.5573, 0.7719>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9553, -1.5621, 1.5204>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0099, -1.5668, 1.7915>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9677, -1.5716, 1.5027>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6192, -1.5764, 0.7468>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7629, -1.5811, -0.2384>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3562, -1.5859, -1.1446>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5287, -1.5906, -1.6896>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4596, -1.5954, -1.7052>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2999, -1.6002, -1.1892>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7308, -1.6049, -0.3050>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6200, -1.6097, 0.6700>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0048, -1.6144, 1.4312>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0794, -1.6192, 1.7425>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8660, -1.6240, 1.5090>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5367, -1.6287, 0.8063>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7249, -1.6335, -0.1441>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3745, -1.6382, -1.0447>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5975, -1.6430, -1.6149>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3617, -1.6478, -1.6788>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2030, -1.6525, -1.2191>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6649, -1.6573, -0.3819>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6054, -1.6620, 0.5698>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0459, -1.6668, 1.3388>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1634, -1.6716, 1.6863>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7650, -1.6763, 1.5063>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4499, -1.6811, 0.8579>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6791, -1.6859, -0.0542>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3837, -1.6906, -0.9442>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6588, -1.6954, -1.5349>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2672, -1.7001, -1.6439>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1045, -1.7049, -1.2400>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5928, -1.7097, -0.4521>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5819, -1.7144, 0.4720>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0781, -1.7192, 1.2436>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2415, -1.7239, 1.6230>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6653, -1.7287, 1.4944>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3593, -1.7335, 0.9011>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6254, -1.7382, 0.0306>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3836, -1.7430, -0.8439>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7122, -1.7477, -1.4500>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1768, -1.7525, -1.6006>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0050, -1.7573, -1.2515>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5148, -1.7620, -0.5149>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5493, -1.7668, 0.3773>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1011, -1.7716, 1.1461>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3128, -1.7763, 1.5528>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5676, -1.7811, 1.4731>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2652, -1.7858, 0.9351>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5640, -1.7906, 0.1095>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3737, -1.7954, -0.7444>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7568, -1.8001, -1.3606>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0914, -1.8049, -1.5487>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9051, -1.8096, -1.2532>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4311, -1.8144, -0.5695>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5076, -1.8192, 0.2865>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1142, -1.8239, 1.0471>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3767, -1.8287, 1.4760>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4726, -1.8334, 1.4421>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1683, -1.8382, 0.9595>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4949, -1.8430, 0.1818>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3537, -1.8477, -0.6465>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7921, -1.8525, -1.2670>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0117, -1.8572, -1.4883>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8054, -1.8620, -1.2444>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3420, -1.8668, -0.6152>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4566, -1.8715, 0.2004>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1168, -1.8763, 0.9469>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4322, -1.8811, 1.3926>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3811, -1.8858, 1.4010>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0689, -1.8906, 0.9734>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4182, -1.8953, 0.2465>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3231, -1.9001, -0.5508>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8172, -1.9049, -1.1697>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0614, -1.9096, -1.4192>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7067, -1.9144, -1.2248>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2479, -1.9191, -0.6511>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3959, -1.9239, 0.1198>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1083, -1.9287, 0.8464>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4786, -1.9334, 1.3028>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2940, -1.9382, 1.3496>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9677, -1.9429, 0.9762>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3339, -1.9477, 0.3027>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2815, -1.9525, -0.4582>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8312, -1.9572, -1.0690>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1269, -1.9620, -1.3412>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6097, -1.9667, -1.1936>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1490, -1.9715, -0.6763>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3254, -1.9763, 0.0458>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0878, -1.9810, 0.7460>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5147, -1.9858, 1.2067>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2123, -1.9906, 1.2873>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8651, -1.9953, 0.9668>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2418, -2.0001, 0.3494>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2282, -2.0048, -0.3696>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8331, -2.0096, -0.9654>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1837, -2.0144, -1.2542>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5151, -2.0191, -1.1499>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0454, -2.0239, -0.6895>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2444, -2.0286, -0.0206>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0544, -2.0334, 0.6465>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5392, -2.0382, 1.1043>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1369, -2.0429, 1.2133>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7617, -2.0477, 0.9441>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1417, -2.0524, 0.3852>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1621, -2.0572, -0.2859>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8214, -2.0620, -0.8590>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2304, -2.0667, -1.1576>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4238, -2.0715, -1.0926>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9372, -2.0763, -0.6894>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1523, -2.0810, -0.0780>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0065, -2.0858, 0.5485>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5507, -2.0905, 0.9954>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0691, -2.0953, 1.1267>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6578, -2.1001, 0.9064>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0331, -2.1048, 0.4084>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0820, -2.1096, -0.2082>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7945, -2.1143, -0.7501>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2655, -2.1191, -1.0504>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3368, -2.1239, -1.0199>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8243, -2.1286, -0.6738>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0476, -2.1334, -0.1249>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9423, -2.1381, 0.4527>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5469, -2.1429, 0.8793>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0104, -2.1477, 1.0256>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5538, -2.1524, 0.8513>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9150, -2.1572, 0.4169>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9856, -2.1619, -0.1379>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7496, -2.1667, -0.6386>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2867, -2.1715, -0.9311>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2550, -2.1762, -0.9292>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7059, -2.1810, -0.6398>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9282, -2.1858, -0.1591>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8583, -2.1905, 0.3596>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5248, -2.1953, 0.7548>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0371, -2.2000, 0.9071>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4497, -2.2048, 0.7753>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7851, -2.2096, 0.4073>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8691, -2.2143, -0.0766>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6825, -2.2191, -0.5237>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2905, -2.2238, -0.7966>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1797, -2.2286, -0.8158>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5804, -2.2334, -0.5828>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7898, -2.2381, -0.1773>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7491, -2.2429, 0.2698>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4790, -2.2476, 0.6189>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0706, -2.2524, 0.7656>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3449, -2.2572, 0.6713>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6389, -2.2619, 0.3741>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7251, -2.2667, -0.0267>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5851, -2.2715, -0.4032>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2712, -2.2762, -0.6401>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1121, -2.2810, -0.6701>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4434, -2.2857, -0.4932>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6228, -2.2905, -0.1737>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6023, -2.2953, 0.1828>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3986, -2.3000, 0.4641>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0847, -2.3048, 0.5876>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2362, -2.3095, 0.5241>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4643, -2.3143, 0.3047>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5351, -2.3191, 0.0076>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4376, -2.3238, -0.2697>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2149, -2.3286, -0.4426>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0536, -2.3333, -0.4659>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2800, -2.3381, -0.3454>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3971, -2.3429, -0.1330>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3797, -2.3476, 0.0943>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2492, -2.3524, 0.2618>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0635, -2.3571, 0.3228>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1066, -2.3619, 0.2728>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2030, -2.3667, 0.1486>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2014, -2.3714, 0.0132>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1180, -2.3762, -0.0647>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0000, 2.1280, 1.1697>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1135, 2.1237, 1.0975>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0806, 2.1195, 0.9974>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0299, 2.1152, 0.9388>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1721, 2.1110, 0.9634>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2860, 2.1067, 1.0784>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3179, 2.1025, 1.2535>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2410, 2.0982, 1.4302>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0679, 2.0939, 1.5427>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1516, 2.0897, 1.5420>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3453, 2.0854, 1.4149>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4434, 2.0812, 1.1924>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4032, 2.0769, 0.9412>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2264, 2.0727, 0.7440>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0389, 2.0684, 0.6711>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3108, 2.0641, 0.7560>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4999, 2.0599, 0.9816>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5385, 2.0556, 1.2826>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4050, 2.0514, 1.5649>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1331, 2.0471, 1.7353>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1962, 2.0429, 1.7324>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4789, 2.0386, 1.5481>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6211, 2.0343, 1.2334>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5705, 2.0301, 0.8839>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3349, 2.0258, 0.6109>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0177, 2.0216, 0.5054>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3787, 2.0173, 0.6081>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6324, 2.0131, 0.8941>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6937, 2.0088, 1.2785>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5360, 2.0046, 1.6418>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2021, 2.0003, 1.8671>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2070, 1.9960, 1.8778>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5633, 1.9918, 1.6635>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7514, 1.9875, 1.2858>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7064, 1.9833, 0.8599>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4358, 1.9790, 0.5201>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0193, 1.9748, 0.3770>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4146, 1.9705, 0.4815>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7285, 1.9662, 0.8069>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8195, 1.9620, 1.2556>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6532, 1.9577, 1.6883>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2761, 1.9535, 1.9673>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1974, 1.9492, 2.0003>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6193, 1.9450, 1.7713>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8548, 1.9407, 1.3468>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8251, 1.9364, 0.8570>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5339, 1.9322, 0.4556>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0679, 1.9279, 0.2714>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4291, 1.9237, 0.3673>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8005, 1.9194, 0.7186>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9263, 1.9152, 1.2192>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7621, 1.9109, 1.7137>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3543, 1.9066, 2.0458>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1728, 1.9024, 2.1075>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6549, 1.8981, 1.8744>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9389, 1.8939, 1.4149>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9317, 1.8896, 0.8701>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6305, 1.8854, 0.4106>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1256, 1.8811, 0.1827>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4272, 1.8768, 0.2623>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8541, 1.8726, 0.6291>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0187, 1.8683, 1.1720>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8648, 1.8641, 1.7226>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4362, 1.8598, 2.1074>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1363, 1.8556, 2.2025>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6742, 1.8513, 1.9738>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0075, 1.8470, 1.4886>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0284, 1.8428, 0.8961>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7258, 1.8385, 0.3814>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1906, 1.8343, 0.1079>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4118, 1.8300, 0.1648>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8926, 1.8258, 0.5387>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0989, 1.8215, 1.1160>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9621, 1.8172, 1.7177>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5209, 1.8130, 2.1546>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0897, 1.8087, 2.2872>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6796, 1.8045, 2.0699>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0629, 1.8002, 1.5669>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1166, 1.7960, 0.9329>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8198, 1.7917, 0.3659>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2618, 1.7875, 0.0450>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3849, 1.7832, 0.0740>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9179, 1.7789, 0.4478>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1684, 1.7747, 1.0526>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0544, 1.7704, 1.7009>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6078, 1.7662, 2.1893>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0346, 1.7619, 2.3625>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6729, 1.7577, 2.1628>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1065, 1.7534, 1.6489>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1969, 1.7491, 0.9791>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9122, 1.7449, 0.3624>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3381, 1.7406, -0.0069>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3479, 1.7364, -0.0103>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9314, 1.7321, 0.3569>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2280, 1.7279, 0.9829>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1418, 1.7236, 1.6736>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6962, 1.7193, 2.2125>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0278, 1.7151, 2.4291>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6555, 1.7108, 2.2522>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1392, 1.7066, 1.7338>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2696, 1.7023, 1.0336>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0026, 1.6981, 0.3699>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4185, 1.6938, -0.0486>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3021, 1.6895, -0.0883>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9341, 1.6853, 0.2665>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2783, 1.6810, 0.9078>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2242, 1.6768, 1.6368>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7854, 1.6725, 2.2252>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0966, 1.6683, 2.4872>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6282, 1.6640, 2.3379>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1617, 1.6597, 1.8208>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3350, 1.6555, 1.0953>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0907, 1.6512, 0.3873>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5023, 1.6470, -0.0807>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2482, 1.6427, -0.1601>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9267, 1.6385, 0.1772>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3196, 1.6342, 0.8283>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3014, 1.6299, 1.5915>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8749, 1.6257, 2.2280>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1708, 1.6214, 2.5372>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5920, 1.6172, 2.4197>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1747, 1.6129, 1.9093>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3930, 1.6087, 1.1633>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1761, 1.6044, 0.4139>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5887, 1.6002, -0.1036>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1873, 1.5959, -0.2255>, 0.06
  texture {
    pigment { rgbf <0.132, 0.403, 0.676, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9100, 1.5916, 0.0894>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3522, 1.5874, 0.7452>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3734, 1.5831, 1.5385>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9640, 1.5789, 2.2215>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2497, 1.5746, 2.5792>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5476, 1.5704, 2.4973>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1784, 1.5661, 1.9986>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4438, 1.5618, 1.2368>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2582, 1.5576, 0.4491>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6770, 1.5533, -0.1176>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1201, 1.5491, -0.2844>, 0.06
  texture {
    pigment { rgbf <0.123, 0.388, 0.657, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8845, 1.5448, 0.0036>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3763, 1.5406, 0.6590>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4397, 1.5363, 1.4784>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0522, 1.5320, 2.2061>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3326, 1.5278, 2.6132>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4956, 1.5235, 2.5703>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1733, 1.5193, 2.0881>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4873, 1.5150, 1.3153>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3368, 1.5108, 0.4923>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7666, 1.5065, -0.1231>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0472, 1.5022, -0.3368>, 0.06
  texture {
    pigment { rgbf <0.110, 0.363, 0.623, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8507, 1.4980, -0.0796>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3922, 1.4937, 0.5706>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5004, 1.4895, 1.4121>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1390, 1.4852, 2.1823>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4187, 1.4810, 2.6393>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4367, 1.4767, 2.6384>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1599, 1.4724, 2.1772>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5234, 1.4682, 1.3978>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4115, 1.4639, 0.5429>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8568, 1.4597, -0.1201>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0307, 1.4554, -0.3825>, 0.06
  texture {
    pigment { rgbf <0.101, 0.346, 0.600, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8092, 1.4512, -0.1598>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3999, 1.4469, 0.4805>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5550, 1.4426, 1.3400>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2237, 1.4384, 2.1505>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5074, 1.4341, 2.6577>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3715, 1.4299, 2.7013>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1383, 1.4256, 2.2653>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5522, 1.4214, 1.4839>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4819, 1.4171, 0.6003>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9470, 1.4128, -0.1091>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1130, 1.4086, -0.4214>, 0.06
  texture {
    pigment { rgbf <0.093, 0.329, 0.576, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7605, 1.4043, -0.2367>, 0.06
  texture {
    pigment { rgbf <0.132, 0.403, 0.676, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3997, 1.4001, 0.3894>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6036, 1.3958, 1.2628>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3060, 1.3916, 2.1111>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5981, 1.3873, 2.6684>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3005, 1.3831, 2.7588>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1089, 1.3788, 2.3519>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5737, 1.3745, 1.5729>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5476, 1.3703, 0.6641>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0367, 1.3660, -0.0903>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1989, 1.3618, -0.4534>, 0.06
  texture {
    pigment { rgbf <0.084, 0.313, 0.553, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7048, 1.3575, -0.3098>, 0.06
  texture {
    pigment { rgbf <0.119, 0.379, 0.646, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3917, 1.3533, 0.2978>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6457, 1.3490, 1.1811>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3854, 1.3447, 2.0645>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6901, 1.3405, 2.6714>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2242, 1.3362, 2.8106>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0721, 1.3320, 2.4365>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5878, 1.3277, 1.6642>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6084, 1.3235, 0.7338>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1254, 1.3192, -0.0638>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2881, 1.3149, -0.4784>, 0.06
  texture {
    pigment { rgbf <0.076, 0.296, 0.530, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6428, 1.3107, -0.3786>, 0.06
  texture {
    pigment { rgbf <0.101, 0.346, 0.600, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3761, 1.3064, 0.2064>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6814, 1.3022, 1.0954>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4615, 1.2979, 2.0111>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7830, 1.2937, 2.6669>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1432, 1.2894, 2.8565>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0281, 1.2851, 2.5187>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5945, 1.2809, 1.7572>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6639, 1.2766, 0.8089>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2125, 1.2724, -0.0300>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3798, 1.2681, -0.4965>, 0.06
  texture {
    pigment { rgbf <0.071, 0.288, 0.519, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5749, 1.2639, -0.4430>, 0.06
  texture {
    pigment { rgbf <0.084, 0.313, 0.553, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3531, 1.2596, 0.1156>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7105, 1.2553, 1.0062>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5337, 1.2511, 1.9513>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8762, 1.2468, 2.6550>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0580, 1.2426, 2.8962>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9774, 1.2383, 2.5978>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5940, 1.2341, 1.8513>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7138, 1.2298, 0.8888>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2975, 1.2255, 0.0108>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4736, 1.2213, -0.5074>, 0.06
  texture {
    pigment { rgbf <0.067, 0.280, 0.507, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5014, 1.2170, -0.5024>, 0.06
  texture {
    pigment { rgbf <0.071, 0.288, 0.519, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3227, 1.2128, 0.0261>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7328, 1.2085, 0.9143>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6018, 1.2043, 1.8854>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9690, 1.2000, 2.6358>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0308, 1.1957, 2.9297>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9202, 1.1915, 2.6737>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5862, 1.1872, 1.9461>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7579, 1.1830, 0.9731>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3799, 1.1787, 0.0585>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5688, 1.1745, -0.5112>, 0.06
  texture {
    pigment { rgbf <0.067, 0.280, 0.507, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4229, 1.1702, -0.5566>, 0.06
  texture {
    pigment { rgbf <0.050, 0.246, 0.461, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2854, 1.1660, -0.0618>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7483, 1.1617, 0.8200>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6654, 1.1574, 1.8140>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0610, 1.1532, 2.6094>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1227, 1.1489, 2.9567>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8569, 1.1447, 2.7457>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5713, 1.1404, 2.0409>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7960, 1.1362, 1.0613>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4594, 1.1319, 0.1126>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6650, 1.1276, -0.5080>, 0.06
  texture {
    pigment { rgbf <0.067, 0.280, 0.507, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3398, 1.1234, -0.6054>, 0.06
  texture {
    pigment { rgbf <0.033, 0.213, 0.415, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2413, 1.1191, -0.1476>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7569, 1.1149, 0.7238>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7242, 1.1106, 1.7373>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1518, 1.1064, 2.5760>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2173, 1.1021, 2.9770>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7878, 1.0978, 2.8136>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5492, 1.0936, 2.1353>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8280, 1.0893, 1.1528>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5354, 1.0851, 0.1730>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7616, 1.0808, -0.4977>, 0.06
  texture {
    pigment { rgbf <0.071, 0.288, 0.519, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2527, 1.0766, -0.6484>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1906, 1.0723, -0.2307>, 0.06
  texture {
    pigment { rgbf <0.132, 0.403, 0.676, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7586, 1.0680, 0.6265>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7777, 1.0638, 1.6559>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2407, 1.0595, 2.5359>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3140, 1.0553, 2.9907>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7134, 1.0510, 2.8770>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5203, 1.0468, 2.2287>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8535, 1.0425, 1.2473>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6077, 1.0382, 0.2392>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8581, 1.0340, -0.4804>, 0.06
  texture {
    pigment { rgbf <0.076, 0.296, 0.530, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1618, 1.0297, -0.6855>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1337, 1.0255, -0.3107>, 0.06
  texture {
    pigment { rgbf <0.114, 0.371, 0.634, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7533, 1.0212, 0.5283>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8259, 1.0170, 1.5701>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3273, 1.0127, 2.4891>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4123, 1.0084, 2.9976>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6340, 1.0042, 2.9355>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4845, 0.9999, 2.3207>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8726, 0.9957, 1.3443>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6757, 0.9914, 0.3109>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9541, 0.9872, -0.4562>, 0.06
  texture {
    pigment { rgbf <0.084, 0.313, 0.553, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0678, 0.9829, -0.7165>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0708, 0.9787, -0.3872>, 0.06
  texture {
    pigment { rgbf <0.101, 0.346, 0.600, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7412, 0.9744, 0.4299>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8683, 0.9701, 1.4804>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4113, 0.9659, 2.4361>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5117, 0.9616, 2.9978>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5501, 0.9574, 2.9888>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4421, 0.9531, 2.4107>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8850, 0.9489, 1.4431>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7392, 0.9446, 0.3878>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0489, 0.9403, -0.4253>, 0.06
  texture {
    pigment { rgbf <0.093, 0.329, 0.576, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0290, 0.9361, -0.7412>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0022, 0.9318, -0.4599>, 0.06
  texture {
    pigment { rgbf <0.080, 0.304, 0.542, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7221, 0.9276, 0.3317>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9049, 0.9233, 1.3872>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4921, 0.9191, 2.3769>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6117, 0.9148, 2.9911>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4620, 0.9105, 3.0367>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3933, 0.9063, 2.4985>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8908, 0.9020, 1.5434>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7978, 0.8978, 0.4694>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1422, 0.8935, -0.3877>, 0.06
  texture {
    pigment { rgbf <0.101, 0.346, 0.600, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1280, 0.8893, -0.7594>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9283, 0.8850, -0.5284>, 0.06
  texture {
    pigment { rgbf <0.063, 0.271, 0.496, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6964, 0.8807, 0.2342>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9353, 0.8765, 1.2910>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5694, 0.8722, 2.3119>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7118, 0.8680, 2.9775>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3701, 0.8637, 3.0790>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3383, 0.8595, 2.5834>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8898, 0.8552, 1.6446>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8512, 0.8509, 0.5555>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2335, 0.8467, -0.3436>, 0.06
  texture {
    pigment { rgbf <0.110, 0.363, 0.623, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2288, 0.8424, -0.7710>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8494, 0.8382, -0.5922>, 0.06
  texture {
    pigment { rgbf <0.037, 0.221, 0.427, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6639, 0.8339, 0.1380>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9595, 0.8297, 1.1923>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6427, 0.8254, 2.2414>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8116, 0.8211, 2.9573>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2749, 0.8169, 3.1153>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2774, 0.8126, 2.6651>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8821, 0.8084, 1.7463>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8992, 0.8041, 0.6456>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3224, 0.7999, -0.2932>, 0.06
  texture {
    pigment { rgbf <0.119, 0.379, 0.646, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3309, 0.7956, -0.7761>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7658, 0.7913, -0.6512>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6249, 0.7871, 0.0434>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9772, 0.7828, 1.0914>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7118, 0.7786, 2.1657>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9105, 0.7743, 2.9303>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1769, 0.7701, 3.1456>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2108, 0.7658, 2.7432>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8677, 0.7616, 1.8480>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9416, 0.7573, 0.7393>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4083, 0.7530, -0.2367>, 0.06
  texture {
    pigment { rgbf <0.132, 0.403, 0.676, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4338, 0.7488, -0.7744>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6780, 0.7445, -0.7050>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5795, 0.7403, -0.0491>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9885, 0.7360, 0.9889>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7762, 0.7318, 2.0851>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0081, 0.7275, 2.8968>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0765, 0.7232, 3.1696>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1388, 0.7190, 2.8173>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8466, 0.7147, 1.9493>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9780, 0.7105, 0.8361>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4910, 0.7062, -0.1744>, 0.06
  texture {
    pigment { rgbf <0.142, 0.417, 0.683, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5370, 0.7020, -0.7661>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5863, 0.6977, -0.7533>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5279, 0.6934, -0.1391>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9931, 0.6892, 0.8853>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8357, 0.6849, 2.0001>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1040, 0.6807, 2.8568>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0260, 0.6764, 3.1872>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0617, 0.6722, 2.8871>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8188, 0.6679, 2.0495>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0083, 0.6636, 0.9357>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5700, 0.6594, -0.1065>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6401, 0.6551, -0.7511>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4911, 0.6509, -0.7959>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4704, 0.6466, -0.2261>, 0.06
  texture {
    pigment { rgbf <0.132, 0.403, 0.676, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9911, 0.6424, 0.7809>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8899, 0.6381, 1.9108>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1976, 0.6338, 2.8105>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1299, 0.6296, 3.1983>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9798, 0.6253, 2.9522>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7846, 0.6211, 2.1484>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0324, 0.6168, 1.0376>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6449, 0.6126, -0.0334>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7427, 0.6083, -0.7295>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3929, 0.6040, -0.8326>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4071, 0.5998, -0.3098>, 0.06
  texture {
    pigment { rgbf <0.119, 0.379, 0.646, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9825, 0.5955, 0.6763>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9387, 0.5913, 1.8178>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2887, 0.5870, 2.7581>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2349, 0.5828, 3.2028>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8935, 0.5785, 3.0123>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7439, 0.5742, 2.2455>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0500, 0.5700, 1.1414>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7155, 0.5657, 0.0448>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8442, 0.5615, -0.7013>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2920, 0.5572, -0.8632>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3383, 0.5530, -0.3897>, 0.06
  texture {
    pigment { rgbf <0.101, 0.346, 0.600, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9672, 0.5487, 0.5719>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9818, 0.5445, 1.7215>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3767, 0.5402, 2.6998>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3406, 0.5359, 3.2007>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8032, 0.5317, 3.0672>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6971, 0.5274, 2.3402>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0613, 0.5232, 1.2465>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7814, 0.5189, 0.1277>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9442, 0.5147, -0.6666>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1890, 0.5104, -0.8875>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2643, 0.5061, -0.4655>, 0.06
  texture {
    pigment { rgbf <0.080, 0.304, 0.542, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9454, 0.5019, 0.4682>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0189, 0.4976, 1.6221>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4613, 0.4934, 2.6358>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4463, 0.4891, 3.1920>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7092, 0.4849, 3.1166>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6441, 0.4806, 2.4323>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0659, 0.4763, 1.3527>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8422, 0.4721, 0.2148>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0424, 0.4678, -0.6256>, 0.06
  texture {
    pigment { rgbf <0.024, 0.197, 0.392, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0841, 0.4636, -0.9055>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1853, 0.4593, -0.5369>, 0.06
  texture {
    pigment { rgbf <0.058, 0.263, 0.484, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9170, 0.4551, 0.3656>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0499, 0.4508, 1.5203>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5422, 0.4465, 2.5664>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5518, 0.4423, 3.1767>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6119, 0.4380, 3.1602>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5853, 0.4338, 2.5213>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0640, 0.4295, 1.4594>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8978, 0.4253, 0.3059>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1381, 0.4210, -0.5784>, 0.06
  texture {
    pigment { rgbf <0.041, 0.230, 0.438, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0221, 0.4167, -0.9170>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1018, 0.4125, -0.6035>, 0.06
  texture {
    pigment { rgbf <0.033, 0.213, 0.415, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8821, 0.4082, 0.2645>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0747, 0.4040, 1.4163>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6188, 0.3997, 2.4918>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6565, 0.3955, 3.1549>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5117, 0.3912, 3.1978>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5209, 0.3869, 2.6068>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0554, 0.3827, 1.5661>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9478, 0.3784, 0.4007>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2311, 0.3742, -0.5252>, 0.06
  texture {
    pigment { rgbf <0.063, 0.271, 0.496, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1291, 0.3699, -0.9219>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0140, 0.3657, -0.6651>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8410, 0.3614, 0.1655>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0931, 0.3571, 1.3106>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6910, 0.3529, 2.4124>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7601, 0.3486, 3.1265>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4091, 0.3444, 3.2294>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4512, 0.3401, 2.6885>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0403, 0.3359, 1.6725>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9920, 0.3316, 0.4986>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3210, 0.3274, -0.4662>, 0.06
  texture {
    pigment { rgbf <0.080, 0.304, 0.542, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2366, 0.3231, -0.9203>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9224, 0.3188, -0.7213>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7938, 0.3146, 0.0689>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1050, 0.3103, 1.2036>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7584, 0.3061, 2.3285>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8620, 0.3018, 3.0917>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3044, 0.2976, 3.2547>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3763, 0.2933, 2.7660>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0186, 0.2890, 1.7780>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0303, 0.2848, 0.5994>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4073, 0.2805, -0.4017>, 0.06
  texture {
    pigment { rgbf <0.097, 0.338, 0.588, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3441, 0.2763, -0.9121>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8272, 0.2720, -0.7720>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7405, 0.2678, -0.0250>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1103, 0.2635, 1.0959>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8207, 0.2592, 2.2403>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9618, 0.2550, 3.0507>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1981, 0.2507, 3.2736>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2967, 0.2465, 2.8389>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9905, 0.2422, 1.8823>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0624, 0.2380, 0.7025>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4897, 0.2337, -0.3318>, 0.06
  texture {
    pigment { rgbf <0.110, 0.363, 0.623, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4512, 0.2294, -0.8972>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7288, 0.2252, -0.8169>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6816, 0.2209, -0.1156>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1091, 0.2167, 0.9877>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8776, 0.2124, 2.1484>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0592, 0.2082, 3.0036>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0907, 0.2039, 3.2860>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2126, 0.1996, 2.9070>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9559, 0.1954, 1.9848>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0882, 0.1911, 0.8076>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5679, 0.1869, -0.2569>, 0.06
  texture {
    pigment { rgbf <0.127, 0.396, 0.669, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5573, 0.1826, -0.8759>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6278, 0.1784, -0.8558>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6171, 0.1741, -0.2026>, 0.06
  texture {
    pigment { rgbf <0.137, 0.410, 0.680, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1013, 0.1698, 0.8797>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9289, 0.1656, 2.0530>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1536, 0.1613, 2.9505>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0175, 0.1571, 3.2918>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1244, 0.1528, 2.9699>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9152, 0.1486, 2.0853>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1077, 0.1443, 0.9143>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6415, 0.1401, -0.1773>, 0.06
  texture {
    pigment { rgbf <0.142, 0.417, 0.683, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6622, 0.1358, -0.8481>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5244, 0.1315, -0.8885>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5473, 0.1273, -0.2857>, 0.06
  texture {
    pigment { rgbf <0.123, 0.388, 0.657, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0869, 0.1230, 0.7722>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9744, 0.1188, 1.9545>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2448, 0.1145, 2.8918>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1261, 0.1103, 3.2911>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0324, 0.1060, 3.0275>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8684, 0.1017, 2.1832>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1206, 0.0975, 1.0220>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7102, 0.0932, -0.0932>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7652, 0.0890, -0.8140>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4191, 0.0847, -0.9149>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4726, 0.0805, -0.3644>, 0.06
  texture {
    pigment { rgbf <0.106, 0.354, 0.611, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0660, 0.0762, 0.6656>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0139, 0.0719, 1.8533>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3324, 0.0677, 2.8275>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2344, 0.0634, 3.2838>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9369, 0.0592, 3.0794>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8156, 0.0549, 2.2782>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1270, 0.0507, 1.1305>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7737, 0.0464, -0.0051>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8661, 0.0421, -0.7736>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3123, 0.0379, -0.9349>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3931, 0.0336, -0.4386>, 0.06
  texture {
    pigment { rgbf <0.089, 0.321, 0.565, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0387, 0.0294, 0.5604>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0472, 0.0251, 1.7499>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4160, 0.0209, 2.7581>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3422, 0.0166, 3.2699>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8385, 0.0123, 3.1255>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7572, 0.0081, 2.3699>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1269, 0.0038, 1.2392>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8319, -0.0004, 0.0868>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9644, -0.0047, -0.7272>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2044, -0.0089, -0.9485>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3093, -0.0132, -0.5078>, 0.06
  texture {
    pigment { rgbf <0.067, 0.280, 0.507, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0051, -0.0175, 0.4571>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0742, -0.0217, 1.6446>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4952, -0.0260, 2.6836>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4490, -0.0302, 3.2496>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7375, -0.0345, 3.1655>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6934, -0.0387, 2.4578>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1201, -0.0430, 1.3477>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8843, -0.0473, 0.1820>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0597, -0.0515, -0.6750>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0960, -0.0558, -0.9554>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2215, -0.0600, -0.5718>, 0.06
  texture {
    pigment { rgbf <0.045, 0.238, 0.450, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9652, -0.0643, 0.3560>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0948, -0.0685, 1.5379>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5698, -0.0728, 2.6046>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5543, -0.0770, 3.2228>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6342, -0.0813, 3.1994>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6244, -0.0856, 2.5417>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1068, -0.0898, 1.4555>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9309, -0.0941, 0.2801>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1516, -0.0983, -0.6171>, 0.06
  texture {
    pigment { rgbf <0.028, 0.205, 0.403, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0127, -0.1026, -0.9558>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1300, -0.1068, -0.6303>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9194, -0.1111, 0.2575>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1088, -0.1154, 1.4302>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6394, -0.1196, 2.5212>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6577, -0.1239, 3.1897>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5292, -0.1281, 3.2269>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5504, -0.1324, 2.6212>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0870, -0.1366, 1.5622>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9714, -0.1409, 0.3808>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2398, -0.1452, -0.5538>, 0.06
  texture {
    pigment { rgbf <0.054, 0.255, 0.473, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1210, -0.1494, -0.9496>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0352, -0.1537, -0.6832>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8677, -0.1579, 0.1621>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1163, -0.1622, 1.3219>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7037, -0.1664, 2.4338>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7589, -0.1707, 3.1504>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4228, -0.1750, 3.2479>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4719, -0.1792, 2.6960>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0608, -0.1835, 1.6675>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0056, -0.1877, 0.4837>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3238, -0.1920, -0.4853>, 0.06
  texture {
    pigment { rgbf <0.076, 0.296, 0.530, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2287, -0.1962, -0.9369>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9375, -0.2005, -0.7301>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8104, -0.2048, 0.0701>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1173, -0.2090, 1.2136>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7625, -0.2133, 2.3429>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8573, -0.2175, 3.1052>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3154, -0.2218, 3.2625>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3891, -0.2260, 2.7658>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0283, -0.2303, 1.7707>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0336, -0.2346, 0.5883>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4035, -0.2388, -0.4120>, 0.06
  texture {
    pigment { rgbf <0.093, 0.329, 0.576, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3352, -0.2431, -0.9176>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8373, -0.2473, -0.7710>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7477, -0.2516, -0.0181>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1116, -0.2558, 1.1056>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8156, -0.2601, 2.2487>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9526, -0.2644, 3.0541>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2076, -0.2686, 3.2704>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3024, -0.2729, 2.8303>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9897, -0.2771, 1.8717>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0550, -0.2814, 0.6942>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4783, -0.2856, -0.3341>, 0.06
  texture {
    pigment { rgbf <0.110, 0.363, 0.623, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4401, -0.2899, -0.8920>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7350, -0.2941, -0.8056>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6799, -0.2984, -0.1021>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0994, -0.3027, 0.9983>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8628, -0.3069, 2.1516>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0444, -0.3112, 2.9975>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0998, -0.3154, 3.2718>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2121, -0.3197, 2.8893>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9451, -0.3239, 1.9698>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0700, -0.3282, 0.8009>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5481, -0.3325, -0.2520>, 0.06
  texture {
    pigment { rgbf <0.127, 0.396, 0.669, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5430, -0.3367, -0.8601>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6311, -0.3410, -0.8338>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6073, -0.3452, -0.1816>, 0.06
  texture {
    pigment { rgbf <0.142, 0.417, 0.683, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0807, -0.3495, 0.8923>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9038, -0.3537, 2.0521>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1324, -0.3580, 2.9355>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0077, -0.3623, 3.2666>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1186, -0.3665, 2.9425>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8947, -0.3708, 2.0648>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0783, -0.3750, 0.9081>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6126, -0.3793, -0.1660>, 0.06
  texture {
    pigment { rgbf <0.142, 0.417, 0.683, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6435, -0.3835, -0.8221>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5259, -0.3878, -0.8555>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5301, -0.3921, -0.2563>, 0.06
  texture {
    pigment { rgbf <0.127, 0.396, 0.669, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0556, -0.3963, 0.7879>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9386, -0.4006, 1.9506>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2161, -0.4048, 2.8684>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1143, -0.4091, 3.2549>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0223, -0.4133, 2.9897>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8387, -0.4176, 2.1562>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0800, -0.4219, 1.0152>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6715, -0.4261, -0.0765>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7411, -0.4304, -0.7781>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4199, -0.4346, -0.8707>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4488, -0.4389, -0.3259>, 0.06
  texture {
    pigment { rgbf <0.114, 0.371, 0.634, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0243, -0.4431, 0.6857>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9669, -0.4474, 1.8475>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2952, -0.4517, 2.7966>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2196, -0.4559, 3.2366>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9237, -0.4602, 3.0307>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7774, -0.4644, 2.2437>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0752, -0.4687, 1.1219>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7245, -0.4729, 0.0161>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8355, -0.4772, -0.7284>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3137, -0.4814, -0.8793>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3637, -0.4857, -0.3900>, 0.06
  texture {
    pigment { rgbf <0.101, 0.346, 0.600, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9868, -0.4900, 0.5859>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9888, -0.4942, 1.7432>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3695, -0.4985, 2.7203>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3231, -0.5027, 3.2120>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8231, -0.5070, 3.0654>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7111, -0.5112, 2.3268>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0638, -0.5155, 1.2276>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7715, -0.5198, 0.1114>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9262, -0.5240, -0.6731>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2075, -0.5283, -0.8812>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2752, -0.5325, -0.4486>, 0.06
  texture {
    pigment { rgbf <0.084, 0.313, 0.553, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9434, -0.5368, 0.4890>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0040, -0.5410, 1.6382>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4386, -0.5453, 2.6398>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4245, -0.5496, 3.1812>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7210, -0.5538, 3.0937>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6401, -0.5581, 2.4054>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0459, -0.5623, 1.3319>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8123, -0.5666, 0.2090>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0130, -0.5708, -0.6126>, 0.06
  texture {
    pigment { rgbf <0.028, 0.205, 0.403, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1018, -0.5751, -0.8766>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1835, -0.5794, -0.5012>, 0.06
  texture {
    pigment { rgbf <0.071, 0.288, 0.519, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8943, -0.5836, 0.3955>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0127, -0.5879, 1.5330>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5023, -0.5921, 2.5557>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5234, -0.5964, 3.1442>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6178, -0.6006, 3.1154>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5647, -0.6049, 2.4790>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0217, -0.6092, 1.4345>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8468, -0.6134, 0.3085>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0953, -0.6177, -0.5470>, 0.06
  texture {
    pigment { rgbf <0.054, 0.255, 0.473, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0028, -0.6219, -0.8654>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0893, -0.6262, -0.5478>, 0.06
  texture {
    pigment { rgbf <0.054, 0.255, 0.473, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8397, -0.6304, 0.3057>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0147, -0.6347, 1.4280>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5602, -0.6390, 2.4681>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6191, -0.6432, 3.1013>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5140, -0.6475, 3.1305>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4852, -0.6517, 2.5473>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9912, -0.6560, 1.5347>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8747, -0.6602, 0.4095>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1730, -0.6645, -0.4768>, 0.06
  texture {
    pigment { rgbf <0.076, 0.296, 0.530, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1059, -0.6688, -0.8477>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9928, -0.6730, -0.5881>, 0.06
  texture {
    pigment { rgbf <0.041, 0.230, 0.438, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7798, -0.6773, 0.2200>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0100, -0.6815, 1.3236>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6122, -0.6858, 2.3775>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7115, -0.6900, 3.0528>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4101, -0.6943, 3.1390>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4019, -0.6985, 2.6102>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9546, -0.7028, 1.6323>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8961, -0.7071, 0.5114>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2456, -0.7113, -0.4022>, 0.06
  texture {
    pigment { rgbf <0.097, 0.338, 0.588, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2071, -0.7156, -0.8237>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8945, -0.7198, -0.6220>, 0.06
  texture {
    pigment { rgbf <0.024, 0.197, 0.392, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7150, -0.7241, 0.1388>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9988, -0.7283, 1.2203>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6580, -0.7326, 2.2844>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8001, -0.7369, 2.9987>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3064, -0.7411, 3.1408>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3154, -0.7454, 2.6672>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9121, -0.7496, 1.7267>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9108, -0.7539, 0.6139>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3128, -0.7581, -0.3236>, 0.06
  texture {
    pigment { rgbf <0.114, 0.371, 0.634, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3060, -0.7624, -0.7934>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7948, -0.7667, -0.6494>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6456, -0.7709, 0.0624>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9811, -0.7752, 1.1187>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6975, -0.7794, 2.1891>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8844, -0.7837, 2.9394>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2035, -0.7879, 3.1359>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2260, -0.7922, 2.7182>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8639, -0.7965, 1.8177>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9188, -0.8007, 0.7164>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3745, -0.8050, -0.2414>, 0.06
  texture {
    pigment { rgbf <0.127, 0.396, 0.669, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4020, -0.8092, -0.7570>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6943, -0.8135, -0.6700>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5718, -0.8177, -0.0089>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9570, -0.8220, 1.0190>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7306, -0.8263, 2.0921>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9642, -0.8305, 2.8753>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1018, -0.8348, 3.1245>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1340, -0.8390, 2.7629>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8103, -0.8433, 1.9046>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9201, -0.8475, 0.8185>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4302, -0.8518, -0.1559>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4948, -0.8561, -0.7147>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5933, -0.8603, -0.6840>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4941, -0.8646, -0.0747>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9267, -0.8688, 0.9219>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7569, -0.8731, 1.9939>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0390, -0.8773, 2.8065>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0018, -0.8816, 3.1065>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0400, -0.8859, 2.8012>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7515, -0.8901, 1.9873>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9147, -0.8944, 0.9197>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4799, -0.8986, -0.0676>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5839, -0.9029, -0.6667>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4924, -0.9071, -0.6912>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4128, -0.9114, -0.1348>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8902, -0.9156, 0.8277>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7766, -0.9199, 1.8949>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1086, -0.9242, 2.7334>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0960, -0.9284, 3.0821>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9444, -0.9327, 2.8330>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6878, -0.9369, 2.0652>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9026, -0.9412, 1.0196>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5231, -0.9454, 0.0231>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6690, -0.9497, -0.6133>, 0.06
  texture {
    pigment { rgbf <0.028, 0.205, 0.403, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3921, -0.9540, -0.6917>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3283, -0.9582, -0.1889>, 0.06
  texture {
    pigment { rgbf <0.137, 0.410, 0.680, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8479, -0.9625, 0.7368>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7894, -0.9667, 1.7956>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1726, -0.9710, 2.6565>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1911, -0.9752, 3.0514>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8477, -0.9795, 2.8580>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6195, -0.9838, 2.1381>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8840, -0.9880, 1.1176>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5598, -0.9923, 0.1158>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7496, -0.9965, -0.5547>, 0.06
  texture {
    pigment { rgbf <0.054, 0.255, 0.473, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2927, -1.0008, -0.6853>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2411, -1.0050, -0.2367>, 0.06
  texture {
    pigment { rgbf <0.132, 0.403, 0.676, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7998, -1.0093, 0.6498>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7954, -1.0136, 1.6964>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2307, -1.0178, 2.5761>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2832, -1.0221, 3.0146>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7503, -1.0263, 2.8762>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5470, -1.0306, 2.2055>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8589, -1.0348, 1.2133>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5898, -1.0391, 0.2100>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8253, -1.0434, -0.4913>, 0.06
  texture {
    pigment { rgbf <0.071, 0.288, 0.519, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1949, -1.0476, -0.6723>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1516, -1.0519, -0.2780>, 0.06
  texture {
    pigment { rgbf <0.123, 0.388, 0.657, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7464, -1.0561, 0.5671>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7946, -1.0604, 1.5980>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2826, -1.0646, 2.4925>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3717, -1.0689, 2.9718>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6528, -1.0732, 2.8874>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4707, -1.0774, 2.2672>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8275, -1.0817, 1.3061>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6130, -1.0859, 0.3051>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8957, -1.0902, -0.4233>, 0.06
  texture {
    pigment { rgbf <0.093, 0.329, 0.576, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0991, -1.0944, -0.6526>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0602, -1.0987, -0.3126>, 0.06
  texture {
    pigment { rgbf <0.114, 0.371, 0.634, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6878, -1.1030, 0.4890>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7869, -1.1072, 1.5008>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3281, -1.1115, 2.4063>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4562, -1.1157, 2.9234>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5555, -1.1200, 2.8917>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3910, -1.1242, 2.3228>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7899, -1.1285, 1.3957>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6292, -1.1327, 0.4008>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9605, -1.1370, -0.3512>, 0.06
  texture {
    pigment { rgbf <0.110, 0.363, 0.623, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0058, -1.1413, -0.6264>, 0.06
  texture {
    pigment { rgbf <0.024, 0.197, 0.392, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9675, -1.1455, -0.3403>, 0.06
  texture {
    pigment { rgbf <0.110, 0.363, 0.623, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6243, -1.1498, 0.4160>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7723, -1.1540, 1.4052>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3669, -1.1583, 2.3179>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5362, -1.1625, 2.8695>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4592, -1.1668, 2.8890>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3082, -1.1711, 2.3720>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7464, -1.1753, 1.4815>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6383, -1.1796, 0.4966>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0194, -1.1838, -0.2752>, 0.06
  texture {
    pigment { rgbf <0.123, 0.388, 0.657, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0844, -1.1881, -0.5938>, 0.06
  texture {
    pigment { rgbf <0.037, 0.221, 0.427, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8738, -1.1923, -0.3610>, 0.06
  texture {
    pigment { rgbf <0.106, 0.354, 0.611, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5563, -1.1966, 0.3485>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7510, -1.2009, 1.3119>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3988, -1.2051, 2.2277>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6113, -1.2094, 2.8104>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3642, -1.2136, 2.8793>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2228, -1.2179, 2.4146>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6971, -1.2221, 1.5631>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6402, -1.2264, 0.5918>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0719, -1.2307, -0.1959>, 0.06
  texture {
    pigment { rgbf <0.137, 0.410, 0.680, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1712, -1.2349, -0.5549>, 0.06
  texture {
    pigment { rgbf <0.054, 0.255, 0.473, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7798, -1.2392, -0.3746>, 0.06
  texture {
    pigment { rgbf <0.101, 0.346, 0.600, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4842, -1.2434, 0.2868>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7231, -1.2477, 1.2213>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4236, -1.2519, 2.1363>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6811, -1.2562, 2.7465>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2711, -1.2605, 2.8626>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1354, -1.2647, 2.4503>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6422, -1.2690, 1.6399>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6350, -1.2732, 0.6859>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1178, -1.2775, -0.1137>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2539, -1.2817, -0.5099>, 0.06
  texture {
    pigment { rgbf <0.067, 0.280, 0.507, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6860, -1.2860, -0.3808>, 0.06
  texture {
    pigment { rgbf <0.101, 0.346, 0.600, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4083, -1.2903, 0.2314>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6886, -1.2945, 1.1340>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4411, -1.2988, 2.0442>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7451, -1.3030, 2.6781>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1805, -1.3073, 2.8391>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0462, -1.3115, 2.4787>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5822, -1.3158, 1.7116>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6226, -1.3200, 0.7785>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1567, -1.3243, -0.0289>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3320, -1.3286, -0.4591>, 0.06
  texture {
    pigment { rgbf <0.080, 0.304, 0.542, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5929, -1.3328, -0.3797>, 0.06
  texture {
    pigment { rgbf <0.101, 0.346, 0.600, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3291, -1.3371, 0.1827>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6477, -1.3413, 1.0505>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4512, -1.3456, 1.9518>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8030, -1.3498, 2.6055>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0929, -1.3541, 2.8086>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9560, -1.3584, 2.4998>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5173, -1.3626, 1.7776>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6030, -1.3669, 0.8689>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1885, -1.3711, 0.0579>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4052, -1.3754, -0.4027>, 0.06
  texture {
    pigment { rgbf <0.097, 0.338, 0.588, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5010, -1.3796, -0.3711>, 0.06
  texture {
    pigment { rgbf <0.106, 0.354, 0.611, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2469, -1.3839, 0.1409>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6006, -1.3882, 0.9713>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4537, -1.3924, 1.8598>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8542, -1.3967, 2.5291>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0089, -1.4009, 2.7715>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8652, -1.4052, 2.5133>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4478, -1.4094, 1.8374>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5762, -1.4137, 0.9565>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2126, -1.4180, 0.1463>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4727, -1.4222, -0.3410>, 0.06
  texture {
    pigment { rgbf <0.110, 0.363, 0.623, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4110, -1.4265, -0.3550>, 0.06
  texture {
    pigment { rgbf <0.106, 0.354, 0.611, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1623, -1.4307, 0.1064>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5475, -1.4350, 0.8970>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4485, -1.4392, 1.7687>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8984, -1.4435, 2.4494>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0710, -1.4478, 2.7277>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7743, -1.4520, 2.5189>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3740, -1.4563, 1.8906>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5423, -1.4605, 1.0409>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2290, -1.4648, 0.2356>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5342, -1.4690, -0.2742>, 0.06
  texture {
    pigment { rgbf <0.123, 0.388, 0.657, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3234, -1.4733, -0.3314>, 0.06
  texture {
    pigment { rgbf <0.110, 0.363, 0.623, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0757, -1.4776, 0.0796>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4886, -1.4818, 0.8282>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4354, -1.4861, 1.6792>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9351, -1.4903, 2.3667>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1461, -1.4946, 2.6774>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6840, -1.4988, 2.5165>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2964, -1.5031, 1.9368>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5013, -1.5074, 1.1213>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2372, -1.5116, 0.3254>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5891, -1.5159, -0.2027>, 0.06
  texture {
    pigment { rgbf <0.137, 0.410, 0.680, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2389, -1.5201, -0.3002>, 0.06
  texture {
    pigment { rgbf <0.119, 0.379, 0.646, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9876, -1.5244, 0.0609>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4242, -1.5286, 0.7654>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4144, -1.5329, 1.5917>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9640, -1.5371, 2.2816>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2159, -1.5414, 2.6208>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5949, -1.5457, 2.5059>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2153, -1.5499, 1.9753>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4533, -1.5542, 1.1971>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2371, -1.5584, 0.4151>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6368, -1.5627, -0.1268>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1581, -1.5669, -0.2614>, 0.06
  texture {
    pigment { rgbf <0.127, 0.396, 0.669, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8986, -1.5712, 0.0507>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3544, -1.5755, 0.7093>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3853, -1.5797, 1.5071>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9845, -1.5840, 2.1945>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2797, -1.5882, 2.5581>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5075, -1.5925, 2.4868>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1313, -1.5967, 2.0058>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3984, -1.6010, 1.2676>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2282, -1.6053, 0.5041>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6767, -1.6095, -0.0469>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0817, -1.6138, -0.2150>, 0.06
  texture {
    pigment { rgbf <0.132, 0.403, 0.676, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8093, -1.6180, 0.0492>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2796, -1.6223, 0.6605>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3480, -1.6265, 1.4259>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9962, -1.6308, 2.1059>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3368, -1.6351, 2.4894>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4225, -1.6393, 2.4591>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0446, -1.6436, 2.0275>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3366, -1.6478, 1.3322>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2103, -1.6521, 0.5917>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7082, -1.6563, 0.0368>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0105, -1.6606, -0.1610>, 0.06
  texture {
    pigment { rgbf <0.142, 0.417, 0.683, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7201, -1.6649, 0.0570>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1999, -1.6691, 0.6196>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3024, -1.6734, 1.3489>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9986, -1.6776, 2.0164>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3865, -1.6819, 2.4148>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3407, -1.6861, 2.4224>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9558, -1.6904, 2.0400>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2681, -1.6947, 1.3901>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1830, -1.6989, 0.6772>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7306, -1.7032, 0.1238>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0549, -1.7074, -0.0992>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6319, -1.7117, 0.0746>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1158, -1.7159, 0.5875>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2481, -1.7202, 1.2768>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9909, -1.7245, 1.9264>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4279, -1.7287, 2.3345>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2629, -1.7330, 2.3764>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8653, -1.7372, 2.0425>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1926, -1.7415, 1.4404>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1458, -1.7457, 0.7600>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7432, -1.7500, 0.2139>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1134, -1.7542, -0.0295>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5451, -1.7585, 0.1025>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0273, -1.7628, 0.5650>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1851, -1.7670, 1.2106>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9727, -1.7713, 1.8367>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4602, -1.7755, 2.2486>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1898, -1.7798, 2.3206>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7735, -1.7840, 2.0341>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1103, -1.7883, 1.4821>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0981, -1.7926, 0.8393>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7448, -1.7968, 0.3065>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1642, -1.8011, 0.0483>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4606, -1.8053, 0.1416>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9347, -1.8096, 0.5532>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1127, -1.8138, 1.1513>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9427, -1.8181, 1.7477>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4821, -1.8224, 2.1571>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1224, -1.8266, 2.2545>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6810, -1.8309, 2.0138>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0208, -1.8351, 1.5141>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0391, -1.8394, 0.9141>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7345, -1.8436, 0.4016>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2060, -1.8479, 0.1347>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3790, -1.8522, 0.1928>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8380, -1.8564, 0.5533>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0303, -1.8607, 1.0999>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9000, -1.8649, 1.6601>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4924, -1.8692, 2.0597>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0618, -1.8734, 2.1771>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5881, -1.8777, 1.9801>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9237, -1.8820, 1.5348>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9674, -1.8862, 0.9835>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7104, -1.8905, 0.4990>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2374, -1.8947, 0.2305>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3012, -1.8990, 0.2578>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7370, -1.9032, 0.5673>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9367, -1.9075, 1.0580>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8425, -1.9118, 1.5744>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4890, -1.9160, 1.9559>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0093, -1.9203, 2.0867>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4951, -1.9245, 1.9309>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8181, -1.9288, 1.5424>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8812, -1.9330, 1.0464>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6703, -1.9373, 0.5987>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2563, -1.9415, 0.3372>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2280, -1.9458, 0.3389>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6312, -1.9501, 0.5976>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8299, -1.9543, 1.0274>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7674, -1.9586, 1.4912>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4692, -1.9628, 1.8446>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0332, -1.9671, 1.9808>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4021, -1.9713, 1.8629>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7019, -1.9756, 1.5339>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7771, -1.9799, 1.1012>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6102, -1.9841, 0.7014>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2598, -1.9884, 0.4575>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1607, -1.9926, 0.4403>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5189, -1.9969, 0.6486>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7062, -2.0011, 1.0111>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6698, -2.0054, 1.4109>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4282, -2.0097, 1.7231>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0632, -2.0139, 1.8542>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3083, -2.0182, 1.7699>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5712, -2.0224, 1.5041>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6483, -2.0267, 1.1458>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5231, -2.0309, 0.8092>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2425, -2.0352, 0.5974>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1002, -2.0395, 0.5706>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3964, -2.0437, 0.7287>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5568, -2.0480, 1.0144>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5386, -2.0522, 1.3331>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3564, -2.0565, 1.5847>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0757, -2.0607, 1.6950>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2112, -2.0650, 1.6383>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4151, -2.0693, 1.4421>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4784, -2.0735, 1.1765>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3913, -2.0778, 0.9285>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1921, -2.0820, 0.7739>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0480, -2.0863, 0.7531>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2503, -2.0905, 0.8609>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3551, -2.0948, 1.0508>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3395, -2.0991, 1.2540>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2228, -2.1033, 1.4038>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0568, -2.1076, 1.4583>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0953, -2.1118, 1.4136>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1815, -2.1161, 1.3025>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1800, -2.1203, 1.1815>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1055, -2.1246, 1.1119>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0000, 2.1280, -1.1697>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1135, 2.1237, -1.2419>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0806, 2.1195, -1.3419>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0299, 2.1152, -1.4005>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1721, 2.1110, -1.3760>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2860, 2.1067, -1.2609>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3179, 2.1025, -1.0859>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2410, 2.0982, -0.9092>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0679, 2.0939, -0.7966>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1516, 2.0897, -0.7974>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3453, 2.0854, -0.9244>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4434, 2.0812, -1.1470>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4032, 2.0769, -1.3981>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2264, 2.0727, -1.5954>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0389, 2.0684, -1.6683>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3108, 2.0641, -1.5833>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4999, 2.0599, -1.3578>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5385, 2.0556, -1.0568>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4050, 2.0514, -0.7744>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1331, 2.0471, -0.6040>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1962, 2.0429, -0.6070>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4789, 2.0386, -0.7913>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6211, 2.0343, -1.1060>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5705, 2.0301, -1.4554>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3349, 2.0258, -1.7285>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0177, 2.0216, -1.8340>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3787, 2.0173, -1.7313>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6324, 2.0131, -1.4453>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6937, 2.0088, -1.0609>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5360, 2.0046, -0.6976>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2021, 2.0003, -0.4723>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2070, 1.9960, -0.4616>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5633, 1.9918, -0.6758>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7514, 1.9875, -1.0536>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7064, 1.9833, -1.4795>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4358, 1.9790, -1.8193>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0193, 1.9748, -1.9624>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4146, 1.9705, -1.8579>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7285, 1.9662, -1.5325>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8195, 1.9620, -1.0838>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6532, 1.9577, -0.6511>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2761, 1.9535, -0.3721>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1974, 1.9492, -0.3390>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6193, 1.9450, -0.5681>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8548, 1.9407, -0.9925>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8251, 1.9364, -1.4823>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5339, 1.9322, -1.8838>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0679, 1.9279, -2.0679>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4291, 1.9237, -1.9720>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8005, 1.9194, -1.6208>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9263, 1.9152, -1.1202>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7621, 1.9109, -0.6256>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3543, 1.9066, -0.2936>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1728, 1.9024, -0.2319>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6549, 1.8981, -0.4650>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9389, 1.8939, -0.9245>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9317, 1.8896, -1.4693>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6305, 1.8854, -1.9288>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1256, 1.8811, -2.1566>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4272, 1.8768, -2.0771>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8541, 1.8726, -1.7103>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0187, 1.8683, -1.1673>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8648, 1.8641, -0.6168>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4362, 1.8598, -0.2320>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1363, 1.8556, -0.1369>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6742, 1.8513, -0.3656>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0075, 1.8470, -0.8508>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0284, 1.8428, -1.4433>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7258, 1.8385, -1.9579>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1906, 1.8343, -2.2315>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4118, 1.8300, -2.1746>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8926, 1.8258, -1.8007>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0989, 1.8215, -1.2233>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9621, 1.8172, -0.6217>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5209, 1.8130, -0.1847>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0897, 1.8087, -0.0522>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6796, 1.8045, -0.2695>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0629, 1.8002, -0.7725>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1166, 1.7960, -1.4065>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8198, 1.7917, -1.9735>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2618, 1.7875, -2.2943>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3849, 1.7832, -2.2653>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9179, 1.7789, -1.8916>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1684, 1.7747, -1.2868>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0544, 1.7704, -0.6385>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6078, 1.7662, -0.1501>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0346, 1.7619, 0.0232>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6729, 1.7577, -0.1766>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1065, 1.7534, -0.6905>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1969, 1.7491, -1.3602>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9122, 1.7449, -1.9769>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3381, 1.7406, -2.3462>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3479, 1.7364, -2.3497>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9314, 1.7321, -1.9825>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2280, 1.7279, -1.3565>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1418, 1.7236, -0.6658>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6962, 1.7193, -0.1268>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0278, 1.7151, 0.0897>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6555, 1.7108, -0.0872>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1392, 1.7066, -0.6056>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2696, 1.7023, -1.3058>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0026, 1.6981, -1.9695>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4185, 1.6938, -2.3880>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3021, 1.6895, -2.4277>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9341, 1.6853, -2.0729>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2783, 1.6810, -1.4315>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2242, 1.6768, -0.7026>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7854, 1.6725, -0.1142>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0966, 1.6683, 0.1479>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6282, 1.6640, -0.0014>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1617, 1.6597, -0.5185>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3350, 1.6555, -1.2441>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0907, 1.6512, -1.9521>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5023, 1.6470, -2.4201>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2482, 1.6427, -2.4994>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9267, 1.6385, -2.1622>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3196, 1.6342, -1.5110>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3014, 1.6299, -0.7479>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8749, 1.6257, -0.1114>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1708, 1.6214, 0.1979>, 0.06
  texture {
    pigment { rgbf <0.137, 0.410, 0.680, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5920, 1.6172, 0.0804>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1747, 1.6129, -0.4300>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3930, 1.6087, -1.1761>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1761, 1.6044, -1.9254>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5887, 1.6002, -2.4430>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1873, 1.5959, -2.5648>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9100, 1.5916, -2.2500>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3522, 1.5874, -1.5942>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3734, 1.5831, -0.8009>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9640, 1.5789, -0.1179>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2497, 1.5746, 0.2398>, 0.06
  texture {
    pigment { rgbf <0.132, 0.403, 0.676, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5476, 1.5704, 0.1579>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1784, 1.5661, -0.3408>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4438, 1.5618, -1.1025>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2582, 1.5576, -1.8902>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6770, 1.5533, -2.4570>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1201, 1.5491, -2.6238>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8845, 1.5448, -2.3357>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3763, 1.5406, -1.6804>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4397, 1.5363, -0.8609>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0522, 1.5320, -0.1333>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3326, 1.5278, 0.2738>, 0.06
  texture {
    pigment { rgbf <0.123, 0.388, 0.657, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4956, 1.5235, 0.2309>, 0.06
  texture {
    pigment { rgbf <0.132, 0.403, 0.676, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1733, 1.5193, -0.2513>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4873, 1.5150, -1.0241>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3368, 1.5108, -1.8471>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7666, 1.5065, -2.4624>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0472, 1.5022, -2.6761>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8507, 1.4980, -2.4189>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3922, 1.4937, -1.7688>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5004, 1.4895, -0.9273>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1390, 1.4852, -0.1571>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4187, 1.4810, 0.3000>, 0.06
  texture {
    pigment { rgbf <0.119, 0.379, 0.646, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4367, 1.4767, 0.2990>, 0.06
  texture {
    pigment { rgbf <0.119, 0.379, 0.646, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1599, 1.4724, -0.1622>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5234, 1.4682, -0.9415>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4115, 1.4639, -1.7965>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8568, 1.4597, -2.4595>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0307, 1.4554, -2.7218>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8092, 1.4512, -2.4992>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3999, 1.4469, -1.8589>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5550, 1.4426, -0.9994>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2237, 1.4384, -0.1889>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5074, 1.4341, 0.3183>, 0.06
  texture {
    pigment { rgbf <0.114, 0.371, 0.634, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3715, 1.4299, 0.3619>, 0.06
  texture {
    pigment { rgbf <0.106, 0.354, 0.611, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1383, 1.4256, -0.0741>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5522, 1.4214, -0.8555>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4819, 1.4171, -1.7390>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9470, 1.4128, -2.4485>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1130, 1.4086, -2.7608>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7605, 1.4043, -2.5761>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3997, 1.4001, -1.9500>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6036, 1.3958, -1.0766>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3060, 1.3916, -0.2283>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5981, 1.3873, 0.3290>, 0.06
  texture {
    pigment { rgbf <0.114, 0.371, 0.634, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3005, 1.3831, 0.4194>, 0.06
  texture {
    pigment { rgbf <0.093, 0.329, 0.576, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1089, 1.3788, 0.0125>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5737, 1.3745, -0.7665>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5476, 1.3703, -1.6752>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0367, 1.3660, -2.4297>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1989, 1.3618, -2.7928>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7048, 1.3575, -2.6491>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3917, 1.3533, -2.0415>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6457, 1.3490, -1.1583>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3854, 1.3447, -0.2749>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6901, 1.3405, 0.3320>, 0.06
  texture {
    pigment { rgbf <0.110, 0.363, 0.623, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2242, 1.3362, 0.4712>, 0.06
  texture {
    pigment { rgbf <0.080, 0.304, 0.542, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0721, 1.3320, 0.0971>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5878, 1.3277, -0.6752>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6084, 1.3235, -1.6056>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1254, 1.3192, -2.4032>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2881, 1.3149, -2.8178>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6428, 1.3107, -2.7180>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3761, 1.3064, -2.1330>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6814, 1.3022, -1.2440>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4615, 1.2979, -0.3283>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7830, 1.2937, 0.3275>, 0.06
  texture {
    pigment { rgbf <0.114, 0.371, 0.634, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1432, 1.2894, 0.5171>, 0.06
  texture {
    pigment { rgbf <0.063, 0.271, 0.496, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0281, 1.2851, 0.1793>, 0.06
  texture {
    pigment { rgbf <0.142, 0.417, 0.683, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5945, 1.2809, -0.5822>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6639, 1.2766, -1.5305>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2125, 1.2724, -2.3694>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3798, 1.2681, -2.8358>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5749, 1.2639, -2.7823>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3531, 1.2596, -2.2237>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7105, 1.2553, -1.3331>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5337, 1.2511, -0.3881>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8762, 1.2468, 0.3156>, 0.06
  texture {
    pigment { rgbf <0.114, 0.371, 0.634, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0580, 1.2426, 0.5569>, 0.06
  texture {
    pigment { rgbf <0.050, 0.246, 0.461, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9774, 1.2383, 0.2585>, 0.06
  texture {
    pigment { rgbf <0.127, 0.396, 0.669, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5940, 1.2341, -0.4881>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7138, 1.2298, -1.4506>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2975, 1.2255, -2.3285>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4736, 1.2213, -2.8468>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5014, 1.2170, -2.8418>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3227, 1.2128, -2.3133>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7328, 1.2085, -1.4251>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6018, 1.2043, -0.4540>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9690, 1.2000, 0.2964>, 0.06
  texture {
    pigment { rgbf <0.119, 0.379, 0.646, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0308, 1.1957, 0.5903>, 0.06
  texture {
    pigment { rgbf <0.037, 0.221, 0.427, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9202, 1.1915, 0.3343>, 0.06
  texture {
    pigment { rgbf <0.110, 0.363, 0.623, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5862, 1.1872, -0.3933>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7579, 1.1830, -1.3663>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3799, 1.1787, -2.2809>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5688, 1.1745, -2.8506>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4229, 1.1702, -2.8960>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2854, 1.1660, -2.4012>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7483, 1.1617, -1.5194>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6654, 1.1574, -0.5254>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0610, 1.1532, 0.2700>, 0.06
  texture {
    pigment { rgbf <0.123, 0.388, 0.657, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1227, 1.1489, 0.6173>, 0.06
  texture {
    pigment { rgbf <0.028, 0.205, 0.403, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8569, 1.1447, 0.4064>, 0.06
  texture {
    pigment { rgbf <0.097, 0.338, 0.588, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5713, 1.1404, -0.2985>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7960, 1.1362, -1.2781>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4594, 1.1319, -2.2267>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6650, 1.1276, -2.8474>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3398, 1.1234, -2.9448>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2413, 1.1191, -2.4869>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7569, 1.1149, -1.6155>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7242, 1.1106, -0.6021>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1518, 1.1064, 0.2367>, 0.06
  texture {
    pigment { rgbf <0.132, 0.403, 0.676, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2173, 1.1021, 0.6377>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7878, 1.0978, 0.4742>, 0.06
  texture {
    pigment { rgbf <0.076, 0.296, 0.530, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5492, 1.0936, -0.2041>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8280, 1.0893, -1.1865>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5354, 1.0851, -2.1664>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7616, 1.0808, -2.8371>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2527, 1.0766, -2.9878>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1906, 1.0723, -2.5700>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7586, 1.0680, -1.7129>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7777, 1.0638, -0.6835>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2407, 1.0595, 0.1965>, 0.06
  texture {
    pigment { rgbf <0.137, 0.410, 0.680, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3140, 1.0553, 0.6514>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7134, 1.0510, 0.5376>, 0.06
  texture {
    pigment { rgbf <0.058, 0.263, 0.484, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5203, 1.0468, -0.1107>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8535, 1.0425, -1.0920>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6077, 1.0382, -2.1002>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8581, 1.0340, -2.8198>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1618, 1.0297, -3.0249>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1337, 1.0255, -2.6501>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7533, 1.0212, -1.8111>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8259, 1.0170, -0.7693>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3273, 1.0127, 0.1498>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4123, 1.0084, 0.6583>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6340, 1.0042, 0.5961>, 0.06
  texture {
    pigment { rgbf <0.037, 0.221, 0.427, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4845, 0.9999, -0.0187>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8726, 0.9957, -0.9951>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6757, 0.9914, -2.0285>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9541, 0.9872, -2.7956>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0678, 0.9829, -3.0559>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0708, 0.9787, -2.7266>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7412, 0.9744, -1.9095>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8683, 0.9701, -0.8590>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4113, 0.9659, 0.0967>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5117, 0.9616, 0.6584>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5501, 0.9574, 0.6495>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4421, 0.9531, 0.0714>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8850, 0.9489, -0.8963>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7392, 0.9446, -1.9516>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0489, 0.9403, -2.7646>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0290, 0.9361, -3.0805>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0022, 0.9318, -2.7993>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7221, 0.9276, -2.0077>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9049, 0.9233, -0.9522>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4921, 0.9191, 0.0375>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6117, 0.9148, 0.6517>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4620, 0.9105, 0.6974>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3933, 0.9063, 0.1591>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8908, 0.9020, -0.7960>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7978, 0.8978, -1.8699>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1422, 0.8935, -2.7270>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1280, 0.8893, -3.0987>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9283, 0.8850, -2.8677>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6964, 0.8807, -2.1051>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9353, 0.8765, -1.0484>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5694, 0.8722, -0.0275>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7118, 0.8680, 0.6382>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3701, 0.8637, 0.7396>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3383, 0.8595, 0.2440>, 0.06
  texture {
    pigment { rgbf <0.127, 0.396, 0.669, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8898, 0.8552, -0.6947>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8512, 0.8509, -1.7839>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2335, 0.8467, -2.6829>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2288, 0.8424, -3.1104>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8494, 0.8382, -2.9316>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6639, 0.8339, -2.2014>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9595, 0.8297, -1.1471>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6427, 0.8254, -0.0980>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8116, 0.8211, 0.6179>, 0.06
  texture {
    pigment { rgbf <0.028, 0.205, 0.403, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2749, 0.8169, 0.7759>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2774, 0.8126, 0.3257>, 0.06
  texture {
    pigment { rgbf <0.114, 0.371, 0.634, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8821, 0.8084, -0.5930>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8992, 0.8041, -1.6938>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3224, 0.7999, -2.6326>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3309, 0.7956, -3.1154>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7658, 0.7913, -2.9906>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6249, 0.7871, -2.2960>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9772, 0.7828, -1.2480>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7118, 0.7786, -0.1737>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9105, 0.7743, 0.5910>, 0.06
  texture {
    pigment { rgbf <0.037, 0.221, 0.427, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1769, 0.7701, 0.8062>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2108, 0.7658, 0.4038>, 0.06
  texture {
    pigment { rgbf <0.097, 0.338, 0.588, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8677, 0.7616, -0.4913>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9416, 0.7573, -1.6001>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4083, 0.7530, -2.5761>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4338, 0.7488, -3.1138>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6780, 0.7445, -3.0444>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5795, 0.7403, -2.3885>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9885, 0.7360, -1.3504>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7762, 0.7318, -0.2542>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0081, 0.7275, 0.5574>, 0.06
  texture {
    pigment { rgbf <0.050, 0.246, 0.461, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0765, 0.7232, 0.8302>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1388, 0.7190, 0.4779>, 0.06
  texture {
    pigment { rgbf <0.076, 0.296, 0.530, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8466, 0.7147, -0.3901>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9780, 0.7105, -1.5033>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4910, 0.7062, -2.5138>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5370, 0.7020, -3.1055>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5863, 0.6977, -3.0927>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5279, 0.6934, -2.4785>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9931, 0.6892, -1.4541>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8357, 0.6849, -0.3393>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1040, 0.6807, 0.5174>, 0.06
  texture {
    pigment { rgbf <0.063, 0.271, 0.496, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0260, 0.6764, 0.8478>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0617, 0.6722, 0.5477>, 0.06
  texture {
    pigment { rgbf <0.054, 0.255, 0.473, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8188, 0.6679, -0.2898>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0083, 0.6636, -1.4037>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5700, 0.6594, -2.4459>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6401, 0.6551, -3.0905>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4911, 0.6509, -3.1353>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4704, 0.6466, -2.5655>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9911, 0.6424, -1.5585>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8899, 0.6381, -0.4285>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1976, 0.6338, 0.4711>, 0.06
  texture {
    pigment { rgbf <0.080, 0.304, 0.542, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1299, 0.6296, 0.8589>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9798, 0.6253, 0.6128>, 0.06
  texture {
    pigment { rgbf <0.028, 0.205, 0.403, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7846, 0.6211, -0.1909>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0324, 0.6168, -1.3018>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6449, 0.6126, -2.3727>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7427, 0.6083, -3.0688>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3929, 0.6040, -3.1720>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4071, 0.5998, -2.6491>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9825, 0.5955, -1.6631>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9387, 0.5913, -0.5215>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2887, 0.5870, 0.4187>, 0.06
  texture {
    pigment { rgbf <0.093, 0.329, 0.576, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2349, 0.5828, 0.8634>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8935, 0.5785, 0.6730>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7439, 0.5742, -0.0939>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0500, 0.5700, -1.1980>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7155, 0.5657, -2.2946>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8442, 0.5615, -3.0406>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2920, 0.5572, -3.2026>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3383, 0.5530, -2.7291>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9672, 0.5487, -1.7675>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9818, 0.5445, -0.6179>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3767, 0.5402, 0.3604>, 0.06
  texture {
    pigment { rgbf <0.106, 0.354, 0.611, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3406, 0.5359, 0.8614>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8032, 0.5317, 0.7278>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6971, 0.5274, 0.0009>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0613, 0.5232, -1.0928>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7814, 0.5189, -2.2117>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9442, 0.5147, -3.0060>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1890, 0.5104, -3.2269>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2643, 0.5061, -2.8049>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9454, 0.5019, -1.8712>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0189, 0.4976, -0.7172>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4613, 0.4934, 0.2964>, 0.06
  texture {
    pigment { rgbf <0.119, 0.379, 0.646, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4463, 0.4891, 0.8527>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7092, 0.4849, 0.7772>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6441, 0.4806, 0.0930>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0659, 0.4763, -0.9867>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8422, 0.4721, -2.1246>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0424, 0.4678, -2.9650>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0841, 0.4636, -3.2449>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1853, 0.4593, -2.8763>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9170, 0.4551, -1.9738>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0499, 0.4508, -0.8191>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5422, 0.4465, 0.2270>, 0.06
  texture {
    pigment { rgbf <0.132, 0.403, 0.676, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5518, 0.4423, 0.8374>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6119, 0.4380, 0.8208>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5853, 0.4338, 0.1820>, 0.06
  texture {
    pigment { rgbf <0.142, 0.417, 0.683, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0640, 0.4295, -0.8800>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8978, 0.4253, -2.0334>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1381, 0.4210, -2.9178>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0221, 0.4167, -3.2564>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1018, 0.4125, -2.9429>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8821, 0.4082, -2.0748>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0747, 0.4040, -0.9231>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6188, 0.3997, 0.1524>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6565, 0.3955, 0.8155>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5117, 0.3912, 0.8585>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5209, 0.3869, 0.2675>, 0.06
  texture {
    pigment { rgbf <0.123, 0.388, 0.657, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0554, 0.3827, -0.7733>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9478, 0.3784, -1.9387>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2311, 0.3742, -2.8646>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1291, 0.3699, -3.2613>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0140, 0.3657, -3.0045>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8410, 0.3614, -2.1739>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0931, 0.3571, -1.0288>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6910, 0.3529, 0.0730>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7601, 0.3486, 0.7871>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4091, 0.3444, 0.8900>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4512, 0.3401, 0.3491>, 0.06
  texture {
    pigment { rgbf <0.110, 0.363, 0.623, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0403, 0.3359, -0.6669>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9920, 0.3316, -1.8408>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3210, 0.3274, -2.8056>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2366, 0.3231, -3.2597>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9224, 0.3188, -3.0607>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7938, 0.3146, -2.2705>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1050, 0.3103, -1.1357>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7584, 0.3061, -0.0109>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8620, 0.3018, 0.7524>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3044, 0.2976, 0.9153>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3763, 0.2933, 0.4266>, 0.06
  texture {
    pigment { rgbf <0.089, 0.321, 0.565, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0186, 0.2890, -0.5614>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0303, 0.2848, -1.7400>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4073, 0.2805, -2.7410>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3441, 0.2763, -3.2514>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8272, 0.2720, -3.1114>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7405, 0.2678, -2.3643>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1103, 0.2635, -1.2435>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8207, 0.2592, -0.0990>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9618, 0.2550, 0.7114>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1981, 0.2507, 0.9342>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2967, 0.2465, 0.4995>, 0.06
  texture {
    pigment { rgbf <0.071, 0.288, 0.519, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9905, 0.2422, -0.4571>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0624, 0.2380, -1.6369>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4897, 0.2337, -2.6712>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4512, 0.2294, -3.2366>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7288, 0.2252, -3.1563>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6816, 0.2209, -2.4549>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1091, 0.2167, -1.3516>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8776, 0.2124, -0.1910>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0592, 0.2082, 0.6642>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0907, 0.2039, 0.9466>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2126, 0.1996, 0.5676>, 0.06
  texture {
    pigment { rgbf <0.045, 0.238, 0.450, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9559, 0.1954, -0.3545>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0882, 0.1911, -1.5318>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5679, 0.1869, -2.5963>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5573, 0.1826, -3.2153>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6278, 0.1784, -3.1951>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6171, 0.1741, -2.5420>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1013, 0.1698, -1.4597>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9289, 0.1656, -0.2864>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1536, 0.1613, 0.6112>, 0.06
  texture {
    pigment { rgbf <0.028, 0.205, 0.403, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0175, 0.1571, 0.9524>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1244, 0.1528, 0.6305>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9152, 0.1486, -0.2541>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1077, 0.1443, -1.4251>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6415, 0.1401, -2.5166>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6622, 0.1358, -3.1875>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5244, 0.1315, -3.2279>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5473, 0.1273, -2.6250>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0869, 0.1230, -1.5672>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9744, 0.1188, -0.3849>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2448, 0.1145, 0.5524>, 0.06
  texture {
    pigment { rgbf <0.054, 0.255, 0.473, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1261, 0.1103, 0.9517>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0324, 0.1060, 0.6881>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8684, 0.1017, -0.1562>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1206, 0.0975, -1.3173>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7102, 0.0932, -2.4326>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7652, 0.0890, -3.1533>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4191, 0.0847, -3.2543>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4726, 0.0805, -2.7038>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0660, 0.0762, -1.6738>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0139, 0.0719, -0.4861>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3324, 0.0677, 0.4882>, 0.06
  texture {
    pigment { rgbf <0.076, 0.296, 0.530, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2344, 0.0634, 0.9444>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9369, 0.0592, 0.7400>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8156, 0.0549, -0.0612>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1270, 0.0507, -1.2089>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7737, 0.0464, -2.3445>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8661, 0.0421, -3.1130>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3123, 0.0379, -3.2743>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3931, 0.0336, -2.7779>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0387, 0.0294, -1.7789>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0472, 0.0251, -0.5895>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4160, 0.0209, 0.4187>, 0.06
  texture {
    pigment { rgbf <0.093, 0.329, 0.576, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3422, 0.0166, 0.9306>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8385, 0.0123, 0.7861>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7572, 0.0081, 0.0305>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1269, 0.0038, -1.1002>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8319, -0.0004, -2.2526>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9644, -0.0047, -3.0666>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2044, -0.0089, -3.2878>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3093, -0.0132, -2.8471>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0051, -0.0175, -1.8823>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0742, -0.0217, -0.6948>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4952, -0.0260, 0.3443>, 0.06
  texture {
    pigment { rgbf <0.110, 0.363, 0.623, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4490, -0.0302, 0.9102>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7375, -0.0345, 0.8261>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6934, -0.0387, 0.1184>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1201, -0.0430, -0.9917>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8843, -0.0473, -2.1574>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0597, -0.0515, -3.0143>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0960, -0.0558, -3.2948>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2215, -0.0600, -2.9112>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9652, -0.0643, -1.9834>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0948, -0.0685, -0.8015>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5698, -0.0728, 0.2652>, 0.06
  texture {
    pigment { rgbf <0.123, 0.388, 0.657, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5543, -0.0770, 0.8834>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6342, -0.0813, 0.8600>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6244, -0.0856, 0.2024>, 0.06
  texture {
    pigment { rgbf <0.137, 0.410, 0.680, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.1068, -0.0898, -0.8839>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9309, -0.0941, -2.0592>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1516, -0.0983, -2.9564>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0127, -0.1026, -3.2952>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1300, -0.1068, -2.9697>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9194, -0.1111, -2.0819>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1088, -0.1154, -0.9092>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6394, -0.1196, 0.1818>, 0.06
  texture {
    pigment { rgbf <0.142, 0.417, 0.683, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6577, -0.1239, 0.8503>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5292, -0.1281, 0.8875>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5504, -0.1324, 0.2819>, 0.06
  texture {
    pigment { rgbf <0.123, 0.388, 0.657, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0870, -0.1366, -0.7771>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9714, -0.1409, -1.9585>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2398, -0.1452, -2.8931>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1210, -0.1494, -3.2890>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0352, -0.1537, -3.0226>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8677, -0.1579, -2.1773>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1163, -0.1622, -1.0174>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7037, -0.1664, 0.0945>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7589, -0.1707, 0.8111>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4228, -0.1750, 0.9085>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4719, -0.1792, 0.3567>, 0.06
  texture {
    pigment { rgbf <0.106, 0.354, 0.611, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0608, -0.1835, -0.6719>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0056, -0.1877, -1.8557>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3238, -0.1920, -2.8247>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2287, -0.1962, -3.2762>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9375, -0.2005, -3.0695>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8104, -0.2048, -2.2693>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1173, -0.2090, -1.1258>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7625, -0.2133, 0.0035>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8573, -0.2175, 0.7658>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3154, -0.2218, 0.9231>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3891, -0.2260, 0.4265>, 0.06
  texture {
    pigment { rgbf <0.089, 0.321, 0.565, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0283, -0.2303, -0.5686>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0336, -0.2346, -1.7511>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4035, -0.2388, -2.7513>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3352, -0.2431, -3.2570>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8373, -0.2473, -3.1103>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7477, -0.2516, -2.3574>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.1116, -0.2558, -1.2338>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8156, -0.2601, -0.0907>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9526, -0.2644, 0.7147>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2076, -0.2686, 0.9311>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3024, -0.2729, 0.4910>, 0.06
  texture {
    pigment { rgbf <0.071, 0.288, 0.519, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9897, -0.2771, -0.4677>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0550, -0.2814, -1.6452>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4783, -0.2856, -2.6735>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4401, -0.2899, -3.2314>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7350, -0.2941, -3.1449>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6799, -0.2984, -2.4414>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0994, -0.3027, -1.3410>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8628, -0.3069, -0.1878>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0444, -0.3112, 0.6581>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0998, -0.3154, 0.9324>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2121, -0.3197, 0.5499>, 0.06
  texture {
    pigment { rgbf <0.054, 0.255, 0.473, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9451, -0.3239, -0.3696>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0700, -0.3282, -1.5384>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5481, -0.3325, -2.5914>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5430, -0.3367, -3.1995>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6311, -0.3410, -3.1731>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6073, -0.3452, -2.5209>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0807, -0.3495, -1.4471>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9038, -0.3537, -0.2873>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1324, -0.3580, 0.5961>, 0.06
  texture {
    pigment { rgbf <0.037, 0.221, 0.427, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0077, -0.3623, 0.9272>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1186, -0.3665, 0.6031>, 0.06
  texture {
    pigment { rgbf <0.033, 0.213, 0.415, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8947, -0.3708, -0.2746>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0783, -0.3750, -1.4313>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6126, -0.3793, -2.5054>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6435, -0.3835, -3.1615>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5259, -0.3878, -3.1949>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5301, -0.3921, -2.5956>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0556, -0.3963, -1.5514>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9386, -0.4006, -0.3888>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2161, -0.4048, 0.5290>, 0.06
  texture {
    pigment { rgbf <0.063, 0.271, 0.496, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1143, -0.4091, 0.9155>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0223, -0.4133, 0.6503>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8387, -0.4176, -0.1832>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0800, -0.4219, -1.3241>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6715, -0.4261, -2.4159>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7411, -0.4304, -3.1175>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4199, -0.4346, -3.2101>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4488, -0.4389, -2.6652>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0243, -0.4431, -1.6537>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9669, -0.4474, -0.4919>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2952, -0.4517, 0.4572>, 0.06
  texture {
    pigment { rgbf <0.084, 0.313, 0.553, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2196, -0.4559, 0.8973>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9237, -0.4602, 0.6913>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7774, -0.4644, -0.0957>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0752, -0.4687, -1.2175>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7245, -0.4729, -2.3233>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8355, -0.4772, -3.0677>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3137, -0.4814, -3.2186>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3637, -0.4857, -2.7294>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9868, -0.4900, -1.7535>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9888, -0.4942, -0.5962>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3695, -0.4985, 0.3809>, 0.06
  texture {
    pigment { rgbf <0.101, 0.346, 0.600, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3231, -0.5027, 0.8727>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8231, -0.5070, 0.7261>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7111, -0.5112, -0.0125>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0638, -0.5155, -1.1118>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7715, -0.5198, -2.2280>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9262, -0.5240, -3.0125>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2075, -0.5283, -3.2206>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2752, -0.5325, -2.7879>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9434, -0.5368, -1.8503>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0040, -0.5410, -0.7011>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4386, -0.5453, 0.3005>, 0.06
  texture {
    pigment { rgbf <0.119, 0.379, 0.646, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4245, -0.5496, 0.8418>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7210, -0.5538, 0.7543>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6401, -0.5581, 0.0660>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0459, -0.5623, -1.0074>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8123, -0.5666, -2.1303>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0130, -0.5708, -2.9519>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1018, -0.5751, -3.2160>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1835, -0.5794, -2.8406>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8943, -0.5836, -1.9438>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0127, -0.5879, -0.8064>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5023, -0.5921, 0.2163>, 0.06
  texture {
    pigment { rgbf <0.132, 0.403, 0.676, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5234, -0.5964, 0.8049>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6178, -0.6006, 0.7761>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5647, -0.6049, 0.1396>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <2.0217, -0.6092, -0.9049>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8468, -0.6134, -2.0308>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0953, -0.6177, -2.8864>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0028, -0.6219, -3.2048>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0893, -0.6262, -2.8872>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8397, -0.6304, -2.0336>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0147, -0.6347, -0.9114>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5602, -0.6390, 0.1287>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6191, -0.6432, 0.7620>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5140, -0.6475, 0.7912>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4852, -0.6517, 0.2080>, 0.06
  texture {
    pigment { rgbf <0.137, 0.410, 0.680, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9912, -0.6560, -0.8046>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8747, -0.6602, -1.9299>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1730, -0.6645, -2.8162>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1059, -0.6688, -3.1871>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9928, -0.6730, -2.9275>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7798, -0.6773, -2.1193>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-2.0100, -0.6815, -1.0158>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6122, -0.6858, 0.0382>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7115, -0.6900, 0.7134>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4101, -0.6943, 0.7996>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4019, -0.6985, 0.2708>, 0.06
  texture {
    pigment { rgbf <0.123, 0.388, 0.657, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9546, -0.7028, -0.7071>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8961, -0.7071, -1.8280>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2456, -0.7113, -2.7416>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2071, -0.7156, -3.1630>, 0.06
  texture {
    pigment { rgbf <0.421, 0.676, 0.819, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8945, -0.7198, -2.9614>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7150, -0.7241, -2.2006>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9988, -0.7283, -1.1190>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6580, -0.7326, -0.0550>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8001, -0.7369, 0.6593>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3064, -0.7411, 0.8014>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3154, -0.7454, 0.3278>, 0.06
  texture {
    pigment { rgbf <0.114, 0.371, 0.634, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9121, -0.7496, -0.6126>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9108, -0.7539, -1.7255>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3128, -0.7581, -2.6630>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3060, -0.7624, -3.1327>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7948, -0.7667, -2.9887>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6456, -0.7709, -2.2770>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9811, -0.7752, -1.2207>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6975, -0.7794, -0.1503>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8844, -0.7837, 0.6001>, 0.06
  texture {
    pigment { rgbf <0.033, 0.213, 0.415, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2035, -0.7879, 0.7966>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2260, -0.7922, 0.3788>, 0.06
  texture {
    pigment { rgbf <0.101, 0.346, 0.600, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8639, -0.7965, -0.5217>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9188, -0.8007, -1.6230>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3745, -0.8050, -2.5808>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4020, -0.8092, -3.0963>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6943, -0.8135, -3.0094>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5718, -0.8177, -2.3483>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9570, -0.8220, -1.3204>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7306, -0.8263, -0.2473>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9642, -0.8305, 0.5359>, 0.06
  texture {
    pigment { rgbf <0.058, 0.263, 0.484, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1018, -0.8348, 0.7851>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1340, -0.8390, 0.4236>, 0.06
  texture {
    pigment { rgbf <0.093, 0.329, 0.576, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8103, -0.8433, -0.4347>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9201, -0.8475, -1.5209>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4302, -0.8518, -2.4953>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4948, -0.8561, -3.0540>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5933, -0.8603, -3.0234>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4941, -0.8646, -2.4141>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.9267, -0.8688, -1.4175>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7569, -0.8731, -0.3455>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0390, -0.8773, 0.4671>, 0.06
  texture {
    pigment { rgbf <0.080, 0.304, 0.542, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0018, -0.8816, 0.7671>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0400, -0.8859, 0.4619>, 0.06
  texture {
    pigment { rgbf <0.080, 0.304, 0.542, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7515, -0.8901, -0.3521>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9147, -0.8944, -1.4196>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4799, -0.8986, -2.4070>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5839, -0.9029, -3.0060>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4924, -0.9071, -3.0306>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4128, -0.9114, -2.4742>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8902, -0.9156, -1.5117>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7766, -0.9199, -0.4445>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1086, -0.9242, 0.3941>, 0.06
  texture {
    pigment { rgbf <0.097, 0.338, 0.588, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0960, -0.9284, 0.7427>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9444, -0.9327, 0.4936>, 0.06
  texture {
    pigment { rgbf <0.071, 0.288, 0.519, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6878, -0.9369, -0.2741>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.9026, -0.9412, -1.3198>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5231, -0.9454, -2.3163>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6690, -0.9497, -2.9526>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3921, -0.9540, -3.0310>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3283, -0.9582, -2.5283>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.8479, -0.9625, -1.6025>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7894, -0.9667, -0.5438>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1726, -0.9710, 0.3171>, 0.06
  texture {
    pigment { rgbf <0.114, 0.371, 0.634, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1911, -0.9752, 0.7120>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8477, -0.9795, 0.5186>, 0.06
  texture {
    pigment { rgbf <0.063, 0.271, 0.496, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6195, -0.9838, -0.2013>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8840, -0.9880, -1.2218>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5598, -0.9923, -2.2236>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7496, -0.9965, -2.8941>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2927, -1.0008, -3.0247>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2411, -1.0050, -2.5761>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7998, -1.0093, -1.6895>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7954, -1.0136, -0.6429>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2307, -1.0178, 0.2367>, 0.06
  texture {
    pigment { rgbf <0.132, 0.403, 0.676, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2832, -1.0221, 0.6752>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7503, -1.0263, 0.5368>, 0.06
  texture {
    pigment { rgbf <0.058, 0.263, 0.484, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5470, -1.0306, -0.1339>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8589, -1.0348, -1.1261>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5898, -1.0391, -2.1294>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8253, -1.0434, -2.8306>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1949, -1.0476, -3.0117>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1516, -1.0519, -2.6174>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7464, -1.0561, -1.7723>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7946, -1.0604, -0.7414>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2826, -1.0646, 0.1531>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3717, -1.0689, 0.6325>, 0.06
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6528, -1.0732, 0.5481>, 0.06
  texture {
    pigment { rgbf <0.054, 0.255, 0.473, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4707, -1.0774, -0.0722>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.8275, -1.0817, -1.0332>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6130, -1.0859, -2.0342>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8957, -1.0902, -2.7627>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0991, -1.0944, -2.9920>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0602, -1.0987, -2.6520>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6878, -1.1030, -1.8504>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7869, -1.1072, -0.8386>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3281, -1.1115, 0.0669>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4562, -1.1157, 0.5840>, 0.06
  texture {
    pigment { rgbf <0.041, 0.230, 0.438, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5555, -1.1200, 0.5524>, 0.06
  texture {
    pigment { rgbf <0.054, 0.255, 0.473, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3910, -1.1242, -0.0166>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7899, -1.1285, -0.9436>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6292, -1.1327, -1.9385>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9605, -1.1370, -2.6905>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0058, -1.1413, -2.9658>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9675, -1.1455, -2.6797>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6243, -1.1498, -1.9234>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7723, -1.1540, -0.9342>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3669, -1.1583, -0.0215>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5362, -1.1625, 0.5301>, 0.06
  texture {
    pigment { rgbf <0.058, 0.263, 0.484, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4592, -1.1668, 0.5496>, 0.06
  texture {
    pigment { rgbf <0.054, 0.255, 0.473, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3082, -1.1711, 0.0327>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.7464, -1.1753, -0.8578>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6383, -1.1796, -1.8428>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0194, -1.1838, -2.6146>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0844, -1.1881, -2.9331>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8738, -1.1923, -2.7004>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5563, -1.1966, -1.9909>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7510, -1.2009, -1.0275>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3988, -1.2051, -0.1116>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6113, -1.2094, 0.4711>, 0.06
  texture {
    pigment { rgbf <0.080, 0.304, 0.542, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3642, -1.2136, 0.5399>, 0.06
  texture {
    pigment { rgbf <0.058, 0.263, 0.484, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2228, -1.2179, 0.0752>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6971, -1.2221, -0.7763>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6402, -1.2264, -1.7476>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0719, -1.2307, -2.5353>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1712, -1.2349, -2.8942>, 0.06
  texture {
    pigment { rgbf <0.409, 0.669, 0.815, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7798, -1.2392, -2.7139>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4842, -1.2434, -2.0525>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.7231, -1.2477, -1.1181>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4236, -1.2519, -0.2030>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6811, -1.2562, 0.4072>, 0.06
  texture {
    pigment { rgbf <0.097, 0.338, 0.588, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2711, -1.2605, 0.5233>, 0.06
  texture {
    pigment { rgbf <0.063, 0.271, 0.496, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1354, -1.2647, 0.1109>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6422, -1.2690, -0.6994>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6350, -1.2732, -1.6535>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1178, -1.2775, -2.4530>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2539, -1.2817, -2.8493>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6860, -1.2860, -2.7202>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4083, -1.2903, -2.1079>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6886, -1.2945, -1.2054>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4411, -1.2988, -0.2952>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7451, -1.3030, 0.3387>, 0.06
  texture {
    pigment { rgbf <0.110, 0.363, 0.623, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1805, -1.3073, 0.4997>, 0.06
  texture {
    pigment { rgbf <0.071, 0.288, 0.519, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0462, -1.3115, 0.1394>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5822, -1.3158, -0.6278>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6226, -1.3200, -1.5609>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1567, -1.3243, -2.3682>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3320, -1.3286, -2.7985>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5929, -1.3328, -2.7190>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3291, -1.3371, -2.1567>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6477, -1.3413, -1.2889>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4512, -1.3456, -0.3875>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8030, -1.3498, 0.2661>, 0.06
  texture {
    pigment { rgbf <0.123, 0.388, 0.657, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0929, -1.3541, 0.4693>, 0.06
  texture {
    pigment { rgbf <0.080, 0.304, 0.542, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9560, -1.3584, 0.1605>, 0.06
  texture {
    pigment { rgbf <0.142, 0.417, 0.683, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5173, -1.3626, -0.5618>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.6030, -1.3669, -1.4705>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1885, -1.3711, -2.2814>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4052, -1.3754, -2.7421>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5010, -1.3796, -2.7105>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2469, -1.3839, -2.1985>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.6006, -1.3882, -1.3681>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4537, -1.3924, -0.4795>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8542, -1.3967, 0.1897>, 0.06
  texture {
    pigment { rgbf <0.137, 0.410, 0.680, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0089, -1.4009, 0.4321>, 0.06
  texture {
    pigment { rgbf <0.089, 0.321, 0.565, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8652, -1.4052, 0.1739>, 0.06
  texture {
    pigment { rgbf <0.142, 0.417, 0.683, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4478, -1.4094, -0.5020>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5762, -1.4137, -1.3828>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2126, -1.4180, -2.1931>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4727, -1.4222, -2.6804>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4110, -1.4265, -2.6944>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1623, -1.4307, -2.2330>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.5475, -1.4350, -1.4423>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4485, -1.4392, -0.5706>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8984, -1.4435, 0.1100>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0710, -1.4478, 0.3883>, 0.06
  texture {
    pigment { rgbf <0.101, 0.346, 0.600, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7743, -1.4520, 0.1796>, 0.06
  texture {
    pigment { rgbf <0.142, 0.417, 0.683, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3740, -1.4563, -0.4487>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5423, -1.4605, -1.2985>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2290, -1.4648, -2.1037>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5342, -1.4690, -2.6136>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3234, -1.4733, -2.6708>, 0.06
  texture {
    pigment { rgbf <0.396, 0.661, 0.810, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0757, -1.4776, -2.2597>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4886, -1.4818, -1.5112>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4354, -1.4861, -0.6602>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9351, -1.4903, 0.0273>, 0.06
  texture {
    pigment { rgbf <0.169, 0.452, 0.701, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1461, -1.4946, 0.3380>, 0.06
  texture {
    pigment { rgbf <0.110, 0.363, 0.623, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6840, -1.4988, 0.1772>, 0.06
  texture {
    pigment { rgbf <0.142, 0.417, 0.683, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2964, -1.5031, -0.4026>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.5013, -1.5074, -1.2181>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2372, -1.5116, -2.0139>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5891, -1.5159, -2.5421>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2389, -1.5201, -2.6396>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9876, -1.5244, -2.2784>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4242, -1.5286, -1.5739>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.4144, -1.5329, -0.7476>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9640, -1.5371, -0.0578>, 0.06
  texture {
    pigment { rgbf <0.179, 0.466, 0.708, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2159, -1.5414, 0.2815>, 0.06
  texture {
    pigment { rgbf <0.123, 0.388, 0.657, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5949, -1.5457, 0.1665>, 0.06
  texture {
    pigment { rgbf <0.142, 0.417, 0.683, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2153, -1.5499, -0.3640>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.4533, -1.5542, -1.1423>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2371, -1.5584, -1.9242>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6368, -1.5627, -2.4662>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1581, -1.5669, -2.6008>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8986, -1.5712, -2.2887>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3544, -1.5755, -1.6301>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3853, -1.5797, -0.8323>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9845, -1.5840, -0.1449>, 0.06
  texture {
    pigment { rgbf <0.190, 0.480, 0.715, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2797, -1.5882, 0.2187>, 0.06
  texture {
    pigment { rgbf <0.132, 0.403, 0.676, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5075, -1.5925, 0.1474>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1313, -1.5967, -0.3336>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3984, -1.6010, -1.0717>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2282, -1.6053, -1.8353>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6767, -1.6095, -2.3862>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0817, -1.6138, -2.5544>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8093, -1.6180, -2.2902>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2796, -1.6223, -1.6789>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3480, -1.6265, -0.9135>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9962, -1.6308, -0.2335>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3368, -1.6351, 0.1500>, 0.06
  texture {
    pigment { rgbf <0.148, 0.424, 0.687, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4225, -1.6393, 0.1197>, 0.06
  texture {
    pigment { rgbf <0.153, 0.431, 0.690, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0446, -1.6436, -0.3118>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.3366, -1.6478, -1.0071>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2103, -1.6521, -1.7477>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7082, -1.6563, -2.3025>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0105, -1.6606, -2.5004>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7201, -1.6649, -2.2824>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1999, -1.6691, -1.7198>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.3024, -1.6734, -0.9905>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9986, -1.6776, -0.3230>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3865, -1.6819, 0.0754>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3407, -1.6861, 0.0830>, 0.06
  texture {
    pigment { rgbf <0.158, 0.438, 0.694, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9558, -1.6904, -0.2994>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.2681, -1.6947, -0.9493>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1830, -1.6989, -1.6621>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7306, -1.7032, -2.2155>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0549, -1.7074, -2.4386>, 0.06
  texture {
    pigment { rgbf <0.384, 0.653, 0.806, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6319, -1.7117, -2.2648>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1158, -1.7159, -1.7519>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.2481, -1.7202, -1.0625>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9909, -1.7245, -0.4129>, 0.06
  texture {
    pigment { rgbf <0.221, 0.521, 0.736, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4279, -1.7287, -0.0048>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2629, -1.7330, 0.0370>, 0.06
  texture {
    pigment { rgbf <0.163, 0.445, 0.698, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8653, -1.7372, -0.2969>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1926, -1.7415, -0.8990>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1458, -1.7457, -1.5794>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7432, -1.7500, -2.1255>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1134, -1.7542, -2.3689>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5451, -1.7585, -2.2369>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0273, -1.7628, -1.7744>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1851, -1.7670, -1.1287>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9727, -1.7713, -0.5027>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4602, -1.7755, -0.0907>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1898, -1.7798, -0.0187>, 0.06
  texture {
    pigment { rgbf <0.174, 0.459, 0.705, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7735, -1.7840, -0.3053>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.1103, -1.7883, -0.8573>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0981, -1.7926, -1.5001>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7448, -1.7968, -2.0328>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1642, -1.8011, -2.2911>, 0.06
  texture {
    pigment { rgbf <0.372, 0.646, 0.802, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4606, -1.8053, -2.1978>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9347, -1.8096, -1.7862>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.1127, -1.8138, -1.1881>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9427, -1.8181, -0.5917>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4821, -1.8224, -0.1823>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1224, -1.8266, -0.0848>, 0.06
  texture {
    pigment { rgbf <0.184, 0.473, 0.712, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6810, -1.8309, -0.3256>, 0.06
  texture {
    pigment { rgbf <0.210, 0.507, 0.729, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0208, -1.8351, -0.8253>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <1.0391, -1.8394, -1.4253>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7345, -1.8436, -1.9377>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2060, -1.8479, -2.2047>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3790, -1.8522, -2.1466>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8380, -1.8564, -1.7860>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-1.0303, -1.8607, -1.2395>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9000, -1.8649, -0.6793>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4924, -1.8692, -0.2797>, 0.06
  texture {
    pigment { rgbf <0.205, 0.500, 0.726, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0618, -1.8734, -0.1623>, 0.06
  texture {
    pigment { rgbf <0.195, 0.487, 0.719, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5881, -1.8777, -0.3593>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9237, -1.8820, -0.8045>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.9674, -1.8862, -1.3559>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7104, -1.8905, -1.8404>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2374, -1.8947, -2.1089>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3012, -1.8990, -2.0816>, 0.06
  texture {
    pigment { rgbf <0.360, 0.638, 0.798, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7370, -1.9032, -1.7721>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.9367, -1.9075, -1.2814>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8425, -1.9118, -0.7650>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4890, -1.9160, -0.3834>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0093, -1.9203, -0.2527>, 0.06
  texture {
    pigment { rgbf <0.200, 0.493, 0.722, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4951, -1.9245, -0.4085>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8181, -1.9288, -0.7969>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.8812, -1.9330, -1.2930>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6703, -1.9373, -1.7407>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2563, -1.9415, -2.0022>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2280, -1.9458, -2.0005>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6312, -1.9501, -1.7418>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.8299, -1.9543, -1.3120>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7674, -1.9586, -0.8481>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4692, -1.9628, -0.4948>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0332, -1.9671, -0.3586>, 0.06
  texture {
    pigment { rgbf <0.216, 0.514, 0.733, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4021, -1.9713, -0.4765>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7019, -1.9756, -0.8055>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.7771, -1.9799, -1.2382>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6102, -1.9841, -1.6379>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2598, -1.9884, -1.8819>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1607, -1.9926, -1.8991>, 0.06
  texture {
    pigment { rgbf <0.348, 0.630, 0.794, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5189, -1.9969, -1.6908>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.7062, -2.0011, -1.3282>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.6698, -2.0054, -0.9285>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.4282, -2.0097, -0.6163>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0632, -2.0139, -0.4852>, 0.06
  texture {
    pigment { rgbf <0.226, 0.528, 0.740, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3083, -2.0182, -0.5694>, 0.06
  texture {
    pigment { rgbf <0.231, 0.535, 0.743, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5712, -2.0224, -0.8352>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.6483, -2.0267, -1.1936>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.5231, -2.0309, -1.5302>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2425, -2.0352, -1.7420>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.1002, -2.0395, -1.7688>, 0.06
  texture {
    pigment { rgbf <0.336, 0.623, 0.790, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3964, -2.0437, -1.6107>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5568, -2.0480, -1.3250>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.5386, -2.0522, -1.0063>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3564, -2.0565, -0.7547>, 0.06
  texture {
    pigment { rgbf <0.247, 0.556, 0.754, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0757, -2.0607, -0.6443>, 0.06
  texture {
    pigment { rgbf <0.237, 0.542, 0.747, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.2112, -2.0650, -0.7011>, 0.06
  texture {
    pigment { rgbf <0.242, 0.549, 0.751, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4151, -2.0693, -0.8972>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.4784, -2.0735, -1.1629>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.3913, -2.0778, -1.4108>, 0.06
  texture {
    pigment { rgbf <0.299, 0.600, 0.777, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1921, -2.0820, -1.5655>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0480, -2.0863, -1.5862>, 0.06
  texture {
    pigment { rgbf <0.323, 0.615, 0.785, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2503, -2.0905, -1.4785>, 0.06
  texture {
    pigment { rgbf <0.311, 0.607, 0.781, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3551, -2.0948, -1.2886>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.3395, -2.0991, -1.0853>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.2228, -2.1033, -0.9356>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <-0.0568, -2.1076, -0.8810>, 0.06
  texture {
    pigment { rgbf <0.252, 0.563, 0.758, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.0953, -2.1118, -0.9257>, 0.06
  texture {
    pigment { rgbf <0.258, 0.570, 0.761, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1815, -2.1161, -1.0368>, 0.06
  texture {
    pigment { rgbf <0.263, 0.576, 0.765, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1800, -2.1203, -1.1579>, 0.06
  texture {
    pigment { rgbf <0.275, 0.584, 0.769, 0.65> }
    finish { esp_finish }
  }
}

sphere {
  <0.1055, -2.1246, -1.2275>, 0.06
  texture {
    pigment { rgbf <0.287, 0.592, 0.773, 0.65> }
    finish { esp_finish }
  }
}

