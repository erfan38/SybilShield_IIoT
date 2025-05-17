// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title FederatedLearning + ReputationManager Integrated System
/// @notice This is an advanced reputation contract using DFL output

contract ReputationManager {
    struct Reputation {
        uint8 trustScore;       // 0–100
        uint32 numReports;      // Number of total reports
        uint256 lastUpdate;     // Timestamp of last update
        bool blacklisted;       // Node is blacklisted
        bytes32 latestModelHash; // Hash of the latest model submission
    }

    address public admin;
    uint8 public blacklistThreshold = 30;
    mapping(address => Reputation) public reputations;
    mapping(address => bool) public authorizedReporters; // trusted validators

    event ReportSubmitted(address indexed reporter, address indexed node, uint8 severity);
    event Blacklisted(address indexed node);
    event Whitelisted(address indexed node);
    event ModelHashRecorded(address indexed node, bytes32 modelHash);

    modifier onlyAdmin() {
        require(msg.sender == admin, "Not admin");
        _;
    }

    constructor() {
        admin = msg.sender;
    }

    function setReporter(address reporter, bool authorized) external onlyAdmin {
        authorizedReporters[reporter] = authorized;
    }

    function setBlacklistThreshold(uint8 newThreshold) external onlyAdmin {
        require(newThreshold < 100, "Too high");
        blacklistThreshold = newThreshold;
    }

    function getReputation(address node) public view returns (Reputation memory) {
        return reputations[node];
    }

    function isBlacklisted(address node) public view returns (bool) {
        return reputations[node].blacklisted;
    }

    /// @notice Submit signed report from DFL validator
    function reportSybil(address node, uint8 severity, bytes calldata signature) external {
        require(severity >= 1 && severity <= 10, "Invalid severity");
        bytes32 message = prefixed(keccak256(abi.encodePacked(node, severity)));
        address signer = recoverSigner(message, signature);
        require(authorizedReporters[signer], "Not authorized");

        Reputation storage rep = reputations[node];
        rep.numReports += 1;
        rep.trustScore = rep.trustScore > severity ? rep.trustScore - severity : 0;
        rep.lastUpdate = block.timestamp;

        emit ReportSubmitted(signer, node, severity);

        if (!rep.blacklisted && rep.trustScore <= blacklistThreshold) {
            rep.blacklisted = true;
            emit Blacklisted(node);
        }
    }

    /// @notice Record model hash for transparency (optional for DFL traceability)
    function recordModelHash(address node, bytes32 modelHash) external {
        require(!reputations[node].blacklisted, "Node is blacklisted");
        reputations[node].latestModelHash = modelHash;
        emit ModelHashRecorded(node, modelHash);
    }

    /// @notice Whitelist and reset trust of a node
    function whitelistNode(address node) external onlyAdmin {
        reputations[node].blacklisted = false;
        reputations[node].trustScore = 100;
        emit Whitelisted(node);
    }

    /// ===== Signature Recovery =====
    function recoverSigner(bytes32 message, bytes memory sig)
        internal
        pure
        returns (address)
    {
        require(sig.length == 65, "Invalid signature length");
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }
        return ecrecover(message, v, r, s);
    }

    function prefixed(bytes32 hash) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", hash));
    }
}
