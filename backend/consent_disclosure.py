"""Versioned prototype disclosure, shared by the consent API and its audit log."""

DISCLOSURE_VERSION = "2026-11-22-v1"
DISCLOSURE = [
    "When enabled, this system uses household microphones to monitor sounds for possible falls, "
    "unusual sounds and prolonged inactivity. It can miss events or raise false alerts; it does not replace "
    "professional care or emergency services and does not automatically call 911.",
    "Household audio may include conversations and sounds from other people. Audio is processed on the Pi; "
    "raw audio is never uploaded to the server or caregiver dashboard. Any local recordings must be deleted "
    "within 24 hours, or earlier when the local storage limit is reached.",
    "The server stores event information such as event type, room, detection "
    "time, delay and detection scores, "
    "along with device status and account information. Linked caregivers can "
    "view alerts and receive alert emails.",
    "By agreeing, you authorize the initial trusted caregiver shown below to access monitoring information "
    "and manage this device. They may invite additional read-only caregivers "
    "without asking you to agree again. "
    "Additional caregivers cannot change monitoring or manage access.",
    "You may withdraw permission on this page using your private consent code, without a dashboard account. "
    "The physical mute control also lets you pause recording. Giving permission "
    "does not itself resume monitoring.",
    "Withdrawal blocks new monitoring alerts and remote resume requests at the server immediately. The Pi "
    "must stop capture, analysis and subsequent monitoring-data collection when it receives withdrawal. "
    "If it is offline, use its local control to stop monitoring immediately; remote confirmation must wait "
    "until it reconnects. Minimal device check-ins remain available to confirm stopping.",
    "Withdrawal does not automatically delete previously saved alert history. Explain monitoring to other "
    "household members and visitors, obtain their permission where needed, and "
    "pause monitoring when permission "
    "is absent. Keep your consent code private: anyone holding it can give or "
    "withdraw permission for this device.",
]
