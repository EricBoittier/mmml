#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sse_types.h"
#include "sse_defs.h"


#ifdef __SSE2__ /* __SSE2__ */
#include "sse_utils.h"
#endif

typedef struct {
  int i, j, ishift;
} list14_t;

#define IN14 1
#define EX14 2

#define NONE 0
#define EWALD 1
#define EWALD_LOOKUP 2
#define CUT 1
#define VSH 2
#define VSW 3
#define VFSW 4

#define KERNEL_NAME enb_fix_14_sse_in14_vsh_ewald
#define FIX_KERNEL_NAME kernel_enb_fix_14_sse_in14_vsh_ewald
#define INTTYPE IN14
#define VDWTYPE VSH
#define ELECTYPE EWALD
#include "enbfix14_sse.h"
#undef ELECTYPE
#undef VDWTYPE
#undef INTTYPE
#undef FIX_KERNEL_NAME
#undef KERNEL_NAME

#define KERNEL_NAME enb_fix_14_sse_in14_vsw_ewald
#define FIX_KERNEL_NAME kernel_enb_fix_14_sse_in14_vsw_ewald
#define INTTYPE IN14
#define VDWTYPE VSW
#define ELECTYPE EWALD
#include "enbfix14_sse.h"
#undef ELECTYPE
#undef VDWTYPE
#undef INTTYPE
#undef FIX_KERNEL_NAME
#undef KERNEL_NAME

#define KERNEL_NAME enb_fix_14_sse_in14_vfsw_ewald
#define FIX_KERNEL_NAME kernel_enb_fix_14_sse_in14_vfsw_ewald
#define INTTYPE IN14
#define VDWTYPE VFSW
#define ELECTYPE EWALD
#include "enbfix14_sse.h"
#undef ELECTYPE
#undef VDWTYPE
#undef INTTYPE
#undef FIX_KERNEL_NAME
#undef KERNEL_NAME

#define KERNEL_NAME enb_fix_14_sse_in14_vsh_none
#define FIX_KERNEL_NAME kernel_enb_fix_14_sse_in14_vsh_none
#define INTTYPE IN14
#define VDWTYPE VSH
#define ELECTYPE NONE
#include "enbfix14_sse.h"
#undef ELECTYPE
#undef VDWTYPE
#undef INTTYPE
#undef FIX_KERNEL_NAME
#undef KERNEL_NAME

#define KERNEL_NAME enb_fix_14_sse_in14_vsw_none
#define FIX_KERNEL_NAME kernel_enb_fix_14_sse_in14_vsw_none
#define INTTYPE IN14
#define VDWTYPE VSW
#define ELECTYPE NONE
#include "enbfix14_sse.h"
#undef ELECTYPE
#undef VDWTYPE
#undef INTTYPE
#undef FIX_KERNEL_NAME
#undef KERNEL_NAME

#define KERNEL_NAME enb_fix_14_sse_in14_vfsw_none
#define FIX_KERNEL_NAME kernel_enb_fix_14_sse_in14_vfsw_none
#define INTTYPE IN14
#define VDWTYPE VFSW
#define ELECTYPE NONE
#include "enbfix14_sse.h"
#undef ELECTYPE
#undef VDWTYPE
#undef INTTYPE
#undef FIX_KERNEL_NAME
#undef KERNEL_NAME

#define KERNEL_NAME enb_fix_14_sse_in14_none_ewald
#define FIX_KERNEL_NAME kernel_enb_fix_14_sse_in14_none_ewald
#define INTTYPE IN14
#define VDWTYPE NONE
#define ELECTYPE EWALD
#include "enbfix14_sse.h"
#undef ELECTYPE
#undef VDWTYPE
#undef INTTYPE
#undef FIX_KERNEL_NAME
#undef KERNEL_NAME

#define KERNEL_NAME enb_fix_14_sse_ex14_none_ewald
#define FIX_KERNEL_NAME kernel_enb_fix_14_sse_ex14_none_ewald
#define INTTYPE EX14
#define VDWTYPE NONE
#define ELECTYPE EWALD
#include "enbfix14_sse.h"
#undef ELECTYPE
#undef VDWTYPE
#undef INTTYPE
#undef FIX_KERNEL_NAME
#undef KERNEL_NAME

#undef NONE
#undef EWALD
#undef EWALD_LOOKUP
#undef CUT
#undef VSH
#undef VSW
#undef VFSW

#undef EX14
#undef IN14
