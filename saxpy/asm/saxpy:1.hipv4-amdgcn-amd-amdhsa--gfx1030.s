
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001900 <_Z12saxpy_kernelfPKfPfj>:
	s_clause 0x1                                               // 000000001900: BFA10001
	s_load_dword s0, s[4:5], 0x2c                              // 000000001904: F4000002 FA00002C
	s_load_dword s1, s[4:5], 0x18                              // 00000000190C: F4000042 FA000018
	s_waitcnt lgkmcnt(0)                                       // 000000001914: BF8CC07F
	s_and_b32 s0, s0, 0xffff                                   // 000000001918: 8700FF00 0000FFFF
	v_mad_u64_u32 v[0:1], null, s6, s0, v[0:1]                 // 000000001920: D5767D00 04000006
	s_mov_b32 s0, exec_lo                                      // 000000001928: BE80037E
	v_cmpx_gt_u32_e64 s1, v0                                   // 00000000192C: D4D4007E 00020001
	s_cbranch_execz 22                                         // 000000001934: BF880016 <_Z12saxpy_kernelfPKfPfj+0x90>
	s_load_dwordx4 s[0:3], s[4:5], 0x8                         // 000000001938: F4080002 FA000008
	v_mov_b32_e32 v1, 0                                        // 000000001940: 7E020280
	v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 000000001944: D6FF0000 00020082
	s_waitcnt lgkmcnt(0)                                       // 00000000194C: BF8CC07F
	v_add_co_u32 v2, vcc_lo, s0, v0                            // 000000001950: D70F6A02 00020000
	v_add_co_ci_u32_e32 v3, vcc_lo, s1, v1, vcc_lo             // 000000001958: 50060201
	v_add_co_u32 v0, vcc_lo, s2, v0                            // 00000000195C: D70F6A00 00020002
	v_add_co_ci_u32_e32 v1, vcc_lo, s3, v1, vcc_lo             // 000000001964: 50020203
	s_load_dword s0, s[4:5], null                              // 000000001968: F4000002 FA000000
	global_load_dword v2, v[2:3], off                          // 000000001970: DC308000 027D0002
	global_load_dword v3, v[0:1], off                          // 000000001978: DC308000 037D0000
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000001980: BF8C0070
	v_fmac_f32_e32 v3, s0, v2                                  // 000000001984: 56060400
	global_store_dword v[0:1], v3, off                         // 000000001988: DC708000 007D0300
	s_endpgm                                                   // 000000001990: BF810000
	s_code_end                                                 // 000000001994: BF9F0000
	s_code_end                                                 // 000000001998: BF9F0000
	s_code_end                                                 // 00000000199C: BF9F0000
	s_code_end                                                 // 0000000019A0: BF9F0000
	s_code_end                                                 // 0000000019A4: BF9F0000
	s_code_end                                                 // 0000000019A8: BF9F0000
	s_code_end                                                 // 0000000019AC: BF9F0000
	s_code_end                                                 // 0000000019B0: BF9F0000
	s_code_end                                                 // 0000000019B4: BF9F0000
	s_code_end                                                 // 0000000019B8: BF9F0000
	s_code_end                                                 // 0000000019BC: BF9F0000
	s_code_end                                                 // 0000000019C0: BF9F0000
	s_code_end                                                 // 0000000019C4: BF9F0000
	s_code_end                                                 // 0000000019C8: BF9F0000
	s_code_end                                                 // 0000000019CC: BF9F0000
	s_code_end                                                 // 0000000019D0: BF9F0000
	s_code_end                                                 // 0000000019D4: BF9F0000
	s_code_end                                                 // 0000000019D8: BF9F0000
	s_code_end                                                 // 0000000019DC: BF9F0000
	s_code_end                                                 // 0000000019E0: BF9F0000
	s_code_end                                                 // 0000000019E4: BF9F0000
	s_code_end                                                 // 0000000019E8: BF9F0000
	s_code_end                                                 // 0000000019EC: BF9F0000
	s_code_end                                                 // 0000000019F0: BF9F0000
	s_code_end                                                 // 0000000019F4: BF9F0000
	s_code_end                                                 // 0000000019F8: BF9F0000
	s_code_end                                                 // 0000000019FC: BF9F0000
	s_code_end                                                 // 000000001A00: BF9F0000
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
