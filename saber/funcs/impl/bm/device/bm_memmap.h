#ifndef ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BM_MEMMAP_H
#define ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BM_MEMMAP_H

#define ITCM_MEM_START_ADDR            0x00000000
#define ITCM_MEM_SIZE                  0x00080000           // 512KB
#define DTCM_MEM_START_ADDR            0x02000000
#define DTCM_MEM_SIZE                  0x00010000           // 64KB
#define SHARE_MEM_START_ADDR           (DTCM_MEM_START_ADDR + DTCM_MEM_SIZE)
#define SHARE_MEM_SIZE                 0x00010000           // 64KB

#define LOCAL_MEM_ADDRWIDTH            18
#define LOCAL_MEM_START_ADDR           0x04000000
//#define LOCAL_MEM_SIZE                 (1<<LOCAL_MEM_ADDRWIDTH)  // 256KB each

#define BDC_RAM_SIZE                   0x00040000    // 512 bit * 4096 entries
#define BDC_RAM_ADDR                   0x45100000

#define SPI_CTLR_BASE_ADDR_REMAP       0x44000000

#define DDR_CTLR_BASE_ADDR             0x50000000

#define GDMA_ENIGNE_BASE_ADDR          0x60000000
#define BD_ENIGNE_BASE_ADDR            0x60002000
#define CDMA_ENIGNE_BASE_ADDR          0x60003000
#define MINIMAC_BASE_ADDR              0x60005000

#define COUNT_RESERVED_DDR_SWAP        0x1000000
#define COUNT_RESERVED_DDR_INSTR       0x1000000
#define COUNT_RESERVED_DDR_IMAGE_SCALE       0x2000000

#define GLOBAL_MEM_START_ADDR_BDC      0x80000000 + COUNT_RESERVED_DDR_INSTR + COUNT_RESERVED_DDR_SWAP
#define GLOBAL_MEM_START_ADDR_ARM      0x80000000

#define GLOBAL_MEM_BOUNDARY            0x80000000
#define PCIE_SOC_BASE_DISTANCE         0x100000000

#ifdef SOC_MODE
  #define GLOBAL_MEM_START_ADDR_CMD      0x0
  #define GLOBAL_MEM_START_ADDR          0x200000000
#else
  #define GLOBAL_MEM_START_ADDR_CMD      0x0
  #define GLOBAL_MEM_START_ADDR          0x100000000
#endif

#define GLOBAL_MEM_ALIGN_SIZE          (4)

#define SPI_CTLR_BASE_ADDR             0xFFF00000

#define ITCM_MAP_B_START_ADDR          0x6001A000

#define SFU_TABLE_ADDR_OFFSET          GLOBAL_MEM_START_ADDR_BDC + 0

#define SOFT_RESET_REG_ADDR            0x50008004
#define GDMA_SOFT_RESET_BIT            5
#define NPS_SOFT_RESET_BIT             6
#define CHIPLINK_SOFT_RESET_BIT        7

#define SHARE_REG_BASE_ADDR            (0x50008240)


#endif /* ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BM_MEMMAP_H */