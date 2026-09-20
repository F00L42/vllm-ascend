add_op_to_compiled_list()

set(MEGA_PTO_INCLUDE "${OPS_TRANSFORMER_DIR}/third_party/pto-isa/include")
if (NOT EXISTS "${MEGA_PTO_INCLUDE}/pto/pto-inst.hpp")
    message(FATAL_ERROR
        "MEGA requires the pinned PTO submodule: git submodule update --init csrc/third_party/pto-isa")
endif()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        ascend_mega_gdn_mtp_decode_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME AscendMegaGdnMtpDecode
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -I${MEGA_PTO_INCLUDE}
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE ascend_mega_gdn_mtp_decode ACLNNTYPE aclnn)
endif()
