# MEGA MTP source provenance

Source: [xLLM-AI/xllm-ops](https://github.com/xLLM-AI/xllm-ops/tree/fee381637f072a4b3a3d3e320aad327b17e27738),
commit `fee381637f072a4b3a3d3e320aad327b17e27738`.
The Apache 2.0 license is preserved in [LICENSE.xllm-ops](LICENSE.xllm-ops).
Original copyright notices remain in the imported files.

- Host/proto/tiling/kernel: `xllm_ops/mega_gdn_mtp_decode/`.
- Shared PTO implementation: `xllm_ops/mega_gdn_decode/op_kernel/mega_gdn_decode_pto_kernel.h`.
- CPU test oracle: `test/python_test/test_mega_gdn_mtp_decode.py`, adapted in
  [tools/mega_gdn/reference.py](../../../tools/mega_gdn/reference.py).
- PTO headers: separate parent-repository submodule `csrc/third_party/pto-isa`,
  pinned to `781ac0eb2e4d31e0bc2ab953bca9f9036bec7528` (the source project's gitlink).

Local adaptations:

1. Rename the CANN operator, generated ACLNN entry and MTP implementation identifiers
   to `AscendMegaGdnMtpDecode` / `ascend_mega_gdn_mtp_decode` to avoid xLLM package symbol collisions.
2. Register only `ascend910b`; other hardware is outside this migration round.
3. Use the parent build's CMake registration from `op_host/`, its pinned PTO include,
   and the adjacent shared header; remove the source project's broad `-Wno-error`.
4. Add a PyTorch wrapper with shape/dtype/device/contiguity/overlap checks, explicit
   mutation of both state pools, and two fresh outputs. Add symbolic Meta outputs.
5. Keep original arithmetic, tiling, slot indexing, and BF16 gate rounding unchanged.
   One comment spelling is adjusted for the parent repository's codespell check.

This standalone P1 ABI cannot address native vLLM physical state pools. It is not
called by the model. See [current contract](../../../spec/mega-gdn/04-current-contract.md)
and [remote validation](../../../spec/mega-gdn/05-server-round-01.md).
