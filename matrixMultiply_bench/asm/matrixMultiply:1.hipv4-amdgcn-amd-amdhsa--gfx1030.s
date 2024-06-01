
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001900 <_Z12sgemm_kerneljPKfS0_Pf>:
	s_clause 0x1                                               // 000000001900: BFA10001
	s_load_dword s0, s[4:5], 0x2c                              // 000000001904: F4000002 FA00002C
	s_load_dword s8, s[4:5], null                              // 00000000190C: F4000202 FA000000
	s_waitcnt lgkmcnt(0)                                       // 000000001914: BF8CC07F
	s_lshr_b32 s1, s0, 16                                      // 000000001918: 90019000
	s_and_b32 s0, s0, 0xffff                                   // 00000000191C: 8700FF00 0000FFFF
	v_mad_u64_u32 v[2:3], null, s6, s0, v[0:1]                 // 000000001924: D5767D02 04000006
	s_mov_b32 s0, exec_lo                                      // 00000000192C: BE80037E
	v_mad_u64_u32 v[0:1], null, s7, s1, v[1:2]                 // 000000001930: D5767D00 04040207
	v_max_u32_e32 v1, v2, v0                                   // 000000001938: 28020102
	v_cmpx_gt_u32_e64 s8, v1                                   // 00000000193C: D4D4007E 00020208
	s_cbranch_execz 46                                         // 000000001944: BF88002E <_Z12sgemm_kerneljPKfS0_Pf+0x100>
	s_clause 0x1                                               // 000000001948: BFA10001
	s_load_dwordx4 s[0:3], s[4:5], 0x8                         // 00000000194C: F4080002 FA000008
	s_load_dwordx2 s[4:5], s[4:5], 0x18                        // 000000001954: F4040102 FA000018
	v_mul_lo_u32 v6, v2, s8                                    // 00000000195C: D5690006 00001102
	v_mov_b32_e32 v2, 0                                        // 000000001964: 7E040280
	v_mov_b32_e32 v3, v0                                       // 000000001968: 7E060300
	v_mov_b32_e32 v5, 0                                        // 00000000196C: 7E0A0280
	s_mov_b32 s6, 0                                            // 000000001970: BE860380
	s_nop 0                                                    // 000000001974: BF800000
	s_nop 0                                                    // 000000001978: BF800000
	s_nop 0                                                    // 00000000197C: BF800000
	v_add_nc_u32_e32 v1, s6, v6                                // 000000001980: 4A020C06
	v_mov_b32_e32 v4, v2                                       // 000000001984: 7E080302
	s_add_i32 s6, s6, 1                                        // 000000001988: 81068106
	s_cmp_eq_u32 s8, s6                                        // 00000000198C: BF060608
	v_lshlrev_b64 v[7:8], 2, v[1:2]                            // 000000001990: D6FF0007 00020282
	v_lshlrev_b64 v[9:10], 2, v[3:4]                           // 000000001998: D6FF0009 00020682
	v_add_nc_u32_e32 v3, s8, v3                                // 0000000019A0: 4A060608
	s_waitcnt lgkmcnt(0)                                       // 0000000019A4: BF8CC07F
	v_add_co_u32 v7, vcc_lo, s0, v7                            // 0000000019A8: D70F6A07 00020E00
	v_add_co_ci_u32_e32 v8, vcc_lo, s1, v8, vcc_lo             // 0000000019B0: 50101001
	v_add_co_u32 v9, vcc_lo, s2, v9                            // 0000000019B4: D70F6A09 00021202
	v_add_co_ci_u32_e32 v10, vcc_lo, s3, v10, vcc_lo           // 0000000019BC: 50141403
	global_load_dword v1, v[7:8], off                          // 0000000019C0: DC308000 017D0007
	global_load_dword v4, v[9:10], off                         // 0000000019C8: DC308000 047D0009
	s_waitcnt vmcnt(0)                                         // 0000000019D0: BF8C3F70
	v_fmac_f32_e32 v5, v1, v4                                  // 0000000019D4: 560A0901
	s_cbranch_scc0 65513                                       // 0000000019D8: BF84FFE9 <_Z12sgemm_kerneljPKfS0_Pf+0x80>
	v_add_nc_u32_e32 v0, v6, v0                                // 0000000019DC: 4A000106
	v_mov_b32_e32 v1, 0                                        // 0000000019E0: 7E020280
	v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 0000000019E4: D6FF0000 00020082
	v_add_co_u32 v0, vcc_lo, s4, v0                            // 0000000019EC: D70F6A00 00020004
	v_add_co_ci_u32_e32 v1, vcc_lo, s5, v1, vcc_lo             // 0000000019F4: 50020205
	global_store_dword v[0:1], v5, off                         // 0000000019F8: DC708000 007D0500
	s_endpgm                                                   // 000000001A00: BF810000
	s_code_end                                                 // 000000001A04: BF9F0000
	s_code_end                                                 // 000000001A08: BF9F0000
	s_code_end                                                 // 000000001A0C: BF9F0000
	s_code_end                                                 // 000000001A10: BF9F0000
	s_code_end                                                 // 000000001A14: BF9F0000
	s_code_end                                                 // 000000001A18: BF9F0000
	s_code_end                                                 // 000000001A1C: BF9F0000
	s_code_end                                                 // 000000001A20: BF9F0000
	s_code_end                                                 // 000000001A24: BF9F0000
	s_code_end                                                 // 000000001A28: BF9F0000
	s_code_end                                                 // 000000001A2C: BF9F0000
	s_code_end                                                 // 000000001A30: BF9F0000
	s_code_end                                                 // 000000001A34: BF9F0000
	s_code_end                                                 // 000000001A38: BF9F0000
	s_code_end                                                 // 000000001A3C: BF9F0000
	s_code_end                                                 // 000000001A40: BF9F0000
	s_code_end                                                 // 000000001A44: BF9F0000
	s_code_end                                                 // 000000001A48: BF9F0000
	s_code_end                                                 // 000000001A4C: BF9F0000
	s_code_end                                                 // 000000001A50: BF9F0000
	s_code_end                                                 // 000000001A54: BF9F0000
	s_code_end                                                 // 000000001A58: BF9F0000
	s_code_end                                                 // 000000001A5C: BF9F0000
	s_code_end                                                 // 000000001A60: BF9F0000
	s_code_end                                                 // 000000001A64: BF9F0000
	s_code_end                                                 // 000000001A68: BF9F0000
	s_code_end                                                 // 000000001A6C: BF9F0000
	s_code_end                                                 // 000000001A70: BF9F0000
	s_code_end                                                 // 000000001A74: BF9F0000
	s_code_end                                                 // 000000001A78: BF9F0000
	s_code_end                                                 // 000000001A7C: BF9F0000
	s_code_end                                                 // 000000001A80: BF9F0000
	s_code_end                                                 // 000000001A84: BF9F0000
	s_code_end                                                 // 000000001A88: BF9F0000
	s_code_end                                                 // 000000001A8C: BF9F0000
	s_code_end                                                 // 000000001A90: BF9F0000
	s_code_end                                                 // 000000001A94: BF9F0000
	s_code_end                                                 // 000000001A98: BF9F0000
	s_code_end                                                 // 000000001A9C: BF9F0000
	s_code_end                                                 // 000000001AA0: BF9F0000
	s_code_end                                                 // 000000001AA4: BF9F0000
	s_code_end                                                 // 000000001AA8: BF9F0000
	s_code_end                                                 // 000000001AAC: BF9F0000
	s_code_end                                                 // 000000001AB0: BF9F0000
	s_code_end                                                 // 000000001AB4: BF9F0000
	s_code_end                                                 // 000000001AB8: BF9F0000
	s_code_end                                                 // 000000001ABC: BF9F0000
	s_code_end                                                 // 000000001AC0: BF9F0000
	s_code_end                                                 // 000000001AC4: BF9F0000
	s_code_end                                                 // 000000001AC8: BF9F0000
	s_code_end                                                 // 000000001ACC: BF9F0000
	s_code_end                                                 // 000000001AD0: BF9F0000
	s_code_end                                                 // 000000001AD4: BF9F0000
	s_code_end                                                 // 000000001AD8: BF9F0000
	s_code_end                                                 // 000000001ADC: BF9F0000
	s_code_end                                                 // 000000001AE0: BF9F0000
	s_code_end                                                 // 000000001AE4: BF9F0000
	s_code_end                                                 // 000000001AE8: BF9F0000
	s_code_end                                                 // 000000001AEC: BF9F0000
	s_code_end                                                 // 000000001AF0: BF9F0000
	s_code_end                                                 // 000000001AF4: BF9F0000
	s_code_end                                                 // 000000001AF8: BF9F0000
	s_code_end                                                 // 000000001AFC: BF9F0000
