# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 2 — Explainability scaffolding
- PR template for Phase 2 development

## [0.2.0] - 2025-01-25

### 🚀 Phase 2 Complete: Payout Optimization & Explainability

This release completes Phase 2 development, delivering deterministic payout optimization, AI-powered decision explanations, and production-ready infrastructure for transparent vendor payout management.

#### Highlights
- **Deterministic Payout Optimization**: Fixed weights for cost (0.5), speed (0.3), and limits (0.2) scoring
- **AI-Powered Decision Explanations**: Azure OpenAI integration for human-readable payout reasoning
- **CloudEvents Integration**: Complete CloudEvent emission for payout optimization decisions with schema validation
- **Production Infrastructure**: Robust CI/CD workflows with security scanning
- **MCP Integration**: Enhanced Model Context Protocol verbs for explainability features

#### Core Features
- **Payout Optimization Engine**: Advanced scoring and ranking of payment rails (ACH, Wire, RTP, V-Card)
- **Rail Management**: Comprehensive payment rail configuration and limit validation
- **Cost Optimization**: Intelligent cost-based rail selection with speed and limit considerations
- **API Endpoints**: RESTful endpoints for payout optimization and decision explanations
- **Event Processing**: Advanced event handling and CloudEvent emission

#### Quality & Infrastructure
- **Test Coverage**: Comprehensive test suite with payout optimization and API validation
- **Security Hardening**: Enhanced security validation and risk assessment
- **CI/CD Pipeline**: Complete GitHub Actions workflow with security scanning
- **Documentation**: Comprehensive API and contract documentation

### Added
- Deterministic payout optimization with fixed weights for reproducible rail scoring
- AI-powered payout decision explanations with Azure OpenAI integration
- LLM integration for human-readable vendor payout reasoning
- Explainability API endpoints for payout plan decisions
- Decision audit trail with explanations
- CloudEvents integration for payout optimization decisions
- Enhanced MCP verbs for explainability features
- Comprehensive payment rail ranking (ACH, Wire, RTP, V-Card)
- Advanced cost optimization with speed and limit considerations
- Production-ready CI/CD infrastructure

### Changed
- Enhanced payout optimization with deterministic scoring
- Improved rail selection with transparent decision logic
- Streamlined MCP integration for better explainability
- Optimized API performance and accuracy

### Deprecated
- None

### Removed
- None

### Fixed
- Resolved mypy type errors across all modules
- Fixed code formatting and type hint issues
- Enhanced error handling and validation
- Improved code quality and consistency

### Security
- Enhanced security validation for payout optimization decisions
- Comprehensive risk assessment and mitigation
- Secure API endpoints with proper authentication
- Robust payout management security measures

## [Unreleased] — Phase 2

### Added
- AI-powered payout decision explanations
- LLM integration for human-readable vendor payout reasoning
- Explainability API endpoints for payout plan decisions
- Decision audit trail with explanations
- Integration with Azure OpenAI for explanations
- Enhanced MCP verbs for explainability features

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 2024-09-22

### Added
- Initial release
- Health check endpoint at `/health`
- FastAPI application setup
- Basic project structure and documentation
