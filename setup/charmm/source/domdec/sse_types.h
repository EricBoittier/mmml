
typedef struct {
  double x, y, z, q;
#if KEY_LJPME==1
  double c6;
#endif
} xyzq_dp_t;

typedef struct {
  float x, y, z, q;
#if KEY_LJPME==1
  float c6;
#endif
} xyzq_sp_t;
