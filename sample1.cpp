    
    
    float vxi = 0.0f, vyi = 0.0f, vzi = 0.0f;
    for (int j = threadIdx.x; j < count1; j += 2*blockDim.x) {
        float2 dx = {xx1[j] - xxi, xx1[j + blockDim.x] - xxi};
        float2 dy = {yy1[j] - yyi, yy1[j + blockDim.x] - yyi};
        float2 dz = {zz1[j] - zzi, zz1[j + blockDim.x] - zzi};
        float2 dist2 = dx*dx + dy*dy + dz*dz;
        bool check[2] = {dist2.x < fsrrmax2, dist2.y < fsrrmax2};
        if (check[0] || check[1]) {
            float2 rtemp = (dist2 + rsm2)*(dist2 + rsm2)*(dist2 + rsm2);
            float2 mass1_2 = {mass1[j],mass1[j + blockDim.x]};
            float2 sqrt_rtemp = {sqrtf(rtemp.x),sqrtf(rtemp.y)};
            float2 f_over_r = massi*mass1_2*(1.0f/sqrt_rtemp - (ma0 + dist2*(ma1 + dist2*(ma2 + dist2*(ma3 + dist2*(ma4 + dist2*ma5))))));
            float2 vxi_tmp = fcoeff*f_over_r*dx;
            float2 vyi_tmp = fcoeff*f_over_r*dy;
            float2 vzi_tmp = fcoeff*f_over_r*dz;
            vxi += check[0] ? vxi_tmp.x : 0.0f;
            vxi += check[1] ? vxi_tmp.y : 0.0f;
            vyi += check[0] ? vyi_tmp.x : 0.0f;
            vyi += check[1] ? vyi_tmp.y : 0.0f;
            vzi += check[0] ? vzi_tmp.x : 0.0f;
            vzi += check[1] ? vzi_tmp.y : 0.0f;
        }
    }

    float vxi = 0.0f, vyi = 0.0f, vzi = 0.0f;
    for (int j = threadIdx.x; j < count1; j += blockDim.x) {
        float dx = xx1[j] - xxi;
        float dy = yy1[j] - yyi;
        float dz = zz1[j] - zzi;
        float dist2 = dx*dx + dy*dy + dz*dz;
        if (dist2 < fsrrmax2) {
            float rtemp = (dist2 + rsm2)*(dist2 + rsm2)*(dist2 + rsm2);
            float f_over_r = massi*mass1[j]*(1.0f/sqrt(rtemp) - (ma0 + dist2*(ma1 + dist2*(ma2 + dist2*(ma3 + dist2*(ma4 + dist2*ma5))))));
            vxi += fcoeff*f_over_r*dx;
            vyi += fcoeff*f_over_r*dy;
            vzi += fcoeff*f_over_r*dz;
        }
    }
