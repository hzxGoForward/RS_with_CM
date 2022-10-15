#ifndef ReedSolomon_H
#define ReedSolomon_H

#include "GaloisField.h"

#include <assert.h>

// Encoder parameters
struct EncoderParams {
    // Original block count < 256
    int OriginalCount;

    // Recovery block count < 256
    int RecoveryCount;

    // Number of bytes per block (all blocks are the same size in bytes)
    int BlockBytes;
};

// Descriptor for data block
struct DataBlock {
    // Pointer to data received.
    void* Block;

    // Block index.
    // For original data, it will be in the range
    //    [0..(originalCount-1)] inclusive.
    // For recovery data, the first one's Index must be originalCount,
    //    and it will be in the range
    //    [originalCount..(originalCount+recoveryCount-1)] inclusive.
    unsigned char Index;
    // Ignored during encoding, required during decoding.
};


/*
 * Cauchy MDS GF(256) encode
 *
 * This produces a set of recovery blocks that should be transmitted after the
 * original data blocks.
 *
 * It takes in 'originalCount' equal-sized blocks and produces 'recoveryCount'
 * equally-sized recovery blocks.
 *
 * The input 'originals' array allows more natural usage of the library.
 * The output recovery blocks are stored end-to-end in 'recoveryBlocks'.
 * 'recoveryBlocks' should have recoveryCount * blockBytes bytes available.
 *
 * Precondition: originalCount + recoveryCount <= 256
 *
 * When transmitting the data, the block index of the data should be sent,
 * and the recovery block index is also needed.  The decoder should also
 * be provided with the values of originalCount, recoveryCount and blockBytes.
 *
 * Example wire format:
 * [originalCount(1 byte)] [recoveryCount(1 byte)]
 * [blockIndex(1 byte)] [blockData(blockBytes bytes)]
 *
 * Be careful not to mix blocks from different encoders.
 *
 * It is possible to support variable-length data by including the original
 * data length at the front of each message in 2 bytes, such that when it is
 * recovered after a loss the data length is available in the block data and
 * the remaining bytes of padding can be neglected.
 *
 * Returns 0 on success, and any other code indicates failure.
 */

class ReedSolomonEncoder{ 
public:
    ReedSolomonEncoder() = delete;
    static int Encode(
    EncoderParams params,          // Encoder parameters
    DataBlock* originals,          // Array of pointers to original blocks
    void* recoveryBlocks);          // Output recovery blocks end-to-end

    // Compute the value to put in the Index member of DataBlock
    static inline unsigned char GetOriginalBlockIndex(EncoderParams params, int originalBlockIndex)
    {
        assert(originalBlockIndex >= 0 && originalBlockIndex < params.OriginalCount);
        return (unsigned char)(originalBlockIndex);
    }
     // Compute the value to put in the Index member of DataBlock
    static inline unsigned char GetRecoveryBlockIndex(EncoderParams params, int recoveryBlockIndex)
    {
        assert(recoveryBlockIndex >= 0 && recoveryBlockIndex < params.RecoveryCount);
        return (unsigned char)(params.OriginalCount + recoveryBlockIndex);
    }
private:
    // Encode one block.
    // Note: This function does not validate input, use with care.
    static void EncodeOneDataBlock(
        EncoderParams params, // Encoder parameters
        DataBlock* originals,      // Array of pointers to original blocks
        int recoveryBlockIndex,      // Return value from GetRecoveryBlockIndex()
        void* recoveryBlock);        // Output recovery block
};



class ReedSolomonDecoder
{
public:
    /*
    * Cauchy MDS GF(256) decode
    *
    * This recovers the original data from the recovery data in the provided
    * blocks.  There should be 'originalCount' blocks in the provided array.
    * Recovery will always be possible if that many blocks are received.
    *
    * Provide the same values for 'originalCount', 'recoveryCount', and
    * 'blockBytes' used by the encoder.
    *
    * The block Index should be set to the block index of the original data,
    * as described in the DataBlock struct comments above.
    *
    * Recovery blocks will be replaced with original data and the Index
    * will be updated to indicate the original block that was recovered.
    *
    * Returns 0 on success, and any other code indicates failure.
    */
    int Decode(
        EncoderParams params, // Encoder parameters
        DataBlock* blocks);        // Array of 'originalCount' blocks as described above

    // Encode parameters
    EncoderParams Params;

    // Recovery blocks
    DataBlock* Recovery[256];
    int RecoveryCount;

    // Original blocks
    DataBlock* Original[256];
    int OriginalCount;

    // Row indices that were erased
    uint8_t ErasuresIndices[256];

    // Initialize the decoder
    bool Initialize(EncoderParams& params, DataBlock* blocks);

    // Decode m=1 case
    void DecodeM1();

    // Decode for m>1 case
    void Decode();

    // Generate the LU decomposition of the matrix
    void GenerateLDUDecomposition(uint8_t* matrix_L, uint8_t* diag_D, uint8_t* matrix_U);

    
};

#endif // ReedSolomon_H