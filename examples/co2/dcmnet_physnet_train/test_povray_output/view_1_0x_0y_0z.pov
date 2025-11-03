#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White}
camera {orthographic
  right -0.84*x up 0.84*y
  direction 1.00*z
  location <0,0,10.00> look_at <0,0,0>}


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
atom(<  0.00,   0.00,  -1.17>, 0.40, rgb <0.56, 0.56, 0.56>, 0.0, jmol) // #0
atom(<  0.00,   0.00,   0.00>, 0.40, rgb <1.00, 0.05, 0.05>, 0.0, jmol) // #1
atom(<  0.00,   0.00,  -2.34>, 0.40, rgb <1.00, 0.05, 0.05>, 0.0, jmol) // #2
cylinder {<  0.00,   0.00,  -1.17>, <  0.00,   0.00,  -0.58>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{jmol}}}
cylinder {<  0.00,   0.00,   0.00>, <  0.00,   0.00,  -0.58>, Rbond texture{pigment {color rgb <1.00, 0.05, 0.05> transmit 0.0} finish{jmol}}}
cylinder {<  0.00,   0.00,  -1.17>, <  0.00,   0.00,  -1.75>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{jmol}}}
cylinder {<  0.00,   0.00,  -2.34>, <  0.00,   0.00,  -1.75>, Rbond texture{pigment {color rgb <1.00, 0.05, 0.05> transmit 0.0} finish{jmol}}}
// no constraints


// Distributed Charges and ESP
// Added by ase_povray_viz.py


#declare charge_texture = texture {
    pigment { rgbf <1, 1, 1, 0.5> }
    finish {
        ambient 0.3
        diffuse 0.6
        specular 0.4
        roughness 0.01
        phong 0.5
        phong_size 40
    }
}

#declare esp_texture = texture {
    finish {
        ambient 0.4
        diffuse 0.5
        specular 0.2
    }
}

sphere {
  <0.0000, 0.0000, 0.0000>, 0.0600
  texture {
    pigment { rgbf <0.855, 0.915, 0.948, 0.50> }
    finish {
      ambient 0.3
      diffuse 0.6
      specular 0.4
      roughness 0.01
    }
  }
}

sphere {
  <0.0000, 0.0000, 0.0000>, 0.0600
  texture {
    pigment { rgbf <0.976, 0.936, 0.913, 0.50> }
    finish {
      ambient 0.3
      diffuse 0.6
      specular 0.4
      roughness 0.01
    }
  }
}

sphere {
  <0.0000, 0.0000, 0.0000>, 0.0600
  texture {
    pigment { rgbf <0.979, 0.919, 0.884, 0.50> }
    finish {
      ambient 0.3
      diffuse 0.6
      specular 0.4
      roughness 0.01
    }
  }
}

sphere {
  <0.0000, 0.0000, 10.5040>, 0.0600
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.50> }
    finish {
      ambient 0.3
      diffuse 0.6
      specular 0.4
      roughness 0.01
    }
  }
}

sphere {
  <0.0000, 0.0000, 0.7567>, 0.0600
  texture {
    pigment { rgbf <0.966, 0.701, 0.579, 0.50> }
    finish {
      ambient 0.3
      diffuse 0.6
      specular 0.4
      roughness 0.01
    }
  }
}

sphere {
  <0.0000, 0.0000, 1.2099>, 0.0600
  texture {
    pigment { rgbf <0.800, 0.299, 0.265, 0.50> }
    finish {
      ambient 0.3
      diffuse 0.6
      specular 0.4
      roughness 0.01
    }
  }
}

sphere {
  <0.0000, 0.0000, -10.5040>, 0.0600
  texture {
    pigment { rgbf <0.020, 0.188, 0.380, 0.50> }
    finish {
      ambient 0.3
      diffuse 0.6
      specular 0.4
      roughness 0.01
    }
  }
}

sphere {
  <0.0000, 0.0000, -0.7567>, 0.0600
  texture {
    pigment { rgbf <0.966, 0.701, 0.579, 0.50> }
    finish {
      ambient 0.3
      diffuse 0.6
      specular 0.4
      roughness 0.01
    }
  }
}

sphere {
  <0.0000, 0.0000, -1.2099>, 0.0600
  texture {
    pigment { rgbf <0.800, 0.299, 0.265, 0.50> }
    finish {
      ambient 0.3
      diffuse 0.6
      specular 0.4
      roughness 0.01
    }
  }
}

