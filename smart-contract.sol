// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title Federated Reputation Management for Sybil Prevention
 * @dev Combines DFL scores, time-weighted decay, behavioral consistency, peer endorsements, and soulbound tokens
 */

contract FederatedReputation {
    struct Reputation {
        uint256 score;                  // Reputation score (0 to 100)
        uint256 lastUpdate;            // Timestamp of last update
        bytes32 behaviorHash;          // Fingerprint of behavioral pattern
        uint256 endorsements;          // Endorsements from other nodes
        bool isFlagged;                // Flag for Sybil suspicion
    }

    mapping(address => Reputation) public reputations;
    mapping(address => mapping(address => bool)) public hasEndorsed;

    uint256 public decayRate = 1 days;            // Time interval for score decay
    uint256 public decayAmount = 1;              // Score units to decay per period
    uint256 public endorsementWeight = 5;        // Max score gain from an endorsement
    uint256 public penaltyForBadEndorse = 10;    // Penalty if an endorsee is flagged
    uint256 public sybilThreshold = 30;          // Score below which a node is suspicious

    event ReputationUpdated(address indexed node, uint256 newScore);
    event NodeFlagged(address indexed node);
    event NodeEndorsed(address indexed endorser, address indexed endorsee);

    modifier onlyTrusted(address node) {
        require(!reputations[node].isFlagged, "Node is flagged as Sybil");
        _;
    }

    function updateFromDFL(address node, uint256 dflScore, bytes32 behaviorHash) external {
        Reputation storage rep = reputations[node];

        // Apply behavior consistency check
        if (rep.behaviorHash != bytes32(0) && rep.behaviorHash == behaviorHash && node != tx.origin) {
            rep.isFlagged = true;
            emit NodeFlagged(node);
            return;
        }

        // Decay over time
        uint256 elapsed = block.timestamp - rep.lastUpdate;
        uint256 decaySteps = elapsed / decayRate;
        if (decaySteps > 0) {
            uint256 totalDecay = decaySteps * decayAmount;
            if (rep.score > totalDecay) {
                rep.score -= totalDecay;
            } else {
                rep.score = 0;
            }
        }

        // Update score with DFL input
        if (dflScore > rep.score) {
            rep.score = dflScore;
        }

        rep.lastUpdate = block.timestamp;
        rep.behaviorHash = behaviorHash;

        // Flag if below threshold
        if (rep.score < sybilThreshold) {
            rep.isFlagged = true;
            emit NodeFlagged(node);
        }

        emit ReputationUpdated(node, rep.score);
    }

    function endorse(address endorsee) external onlyTrusted(msg.sender) {
        require(!hasEndorsed[msg.sender][endorsee], "Already endorsed");
        hasEndorsed[msg.sender][endorsee] = true;

        Reputation storage rep = reputations[endorsee];
        if (!rep.isFlagged) {
            rep.score += endorsementWeight;
            rep.endorsements += 1;
        } else {
            // Punish the endorser if they vouch for a Sybil
            reputations[msg.sender].score = reputations[msg.sender].score > penaltyForBadEndorse
                ? reputations[msg.sender].score - penaltyForBadEndorse
                : 0;
        }

        emit NodeEndorsed(msg.sender, endorsee);
    }

    function isSybil(address node) external view returns (bool) {
        return reputations[node].isFlagged;
    }

    function getReputation(address node) external view returns (uint256) {
        return reputations[node].score;
    }
}
