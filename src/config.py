import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Behavior descriptions for malware analysis

BEHAVIOR_DESCRIPTIONS = {
    1: """Privacy Stealing - Methods that access or exfiltrate sensitive user data including:
(1) - Accessing Contact Lists – Retrieving the user's contact details from the device's storage.
(2) - Reading SMS Messages – Accessing and potentially forwarding SMS messages to external servers.
(3) - Collecting Location Data – Gathering precise GPS or network-based location information.
(4) - Extracting Phone Numbers – Accessing the device's phone number or identifiers such as IMEI and IMSI.
(5) - Harvesting Call Logs – Reading historical data on incoming, outgoing, or missed calls.
(6) - Intercepting Communications – Monitoring or manipulating SMS or call-based communication.
(7) - Exfiltrating User Data – Sending private information to external servers or networks.
Look for: Permission checks, content provider queries, telephony manager access, location services, file operations targeting private directories.""",

    2: """SMS/CALL Abuse - Methods that manipulate SMS and phone call functionality:
(1) - Sending SMS messages without user consent
(2) - Intercepting/blocking incoming SMS (especially 2FA messages)
(3) - Deleting SMS messages (to hide evidence)
(4) - Making calls without user awareness
(5) - Monitoring call logs
Look for: SMS manager operations, broadcast receivers for SMS/calls, telephony API usage, SMS deletion commands.""",

    3: """Remote Control - Methods enabling C&C server communication and remote command execution:
(1) - Network connections to remote servers
(2) - WebSocket protocol usage
(3) - Command parsing and execution
(4) - Dynamic code loading
(5) - Background service creation
Common commands: sendSms, show_fs_float_window (phishing overlays)
Look for: Socket connections, HTTP clients, WebSocket implementations, service registrations, dynamic loading.""",

    4:   """Bank/Financial Stealing - Methods implementing banking trojan functionality:
(1) - Overlay attacks on banking apps
(2) - Credential theft
(3) - Screen capture during banking sessions
(4) - Banking app detection
Example: Exobot-style phishing windows
Look for: Window overlay APIs, package monitoring, screen capture calls, banking app package names in strings.""",

    5: """Ransom - Methods implementing ransomware behavior:
(1) - File encryption operations
(2) - Screen locking mechanisms
(3) - Payment demand displays
(4) - Bitcoin/cryptocurrency payment processing
Example: SLocker patterns
Look for: Encryption APIs, screen locking calls, file system operations, payment-related strings.""",

    6: """Accessibility Abuse - Methods exploiting accessibility services:
(1) - Accessibility service registration
(2) - Screen content monitoring
(3) - Automated UI interaction
(4) - Silent installation attempts
Example: TOASTAMIGO patterns
Look for: Accessibility service declarations, window content observers, automated click events.""",

    7: """Privilege Escalation - Methods attempting to gain elevated privileges:
(1) - Root exploit attempts
(2) - System file modifications
(3) - Admin privilege requests
(4) - Persistent privilege elevation
Examples: LIBSKIN (right_core.apk), ZNIU (Dirty COW)
Look for: Root checking, system file operations, privilege escalation exploits, admin rights requests.""",

    8: """Stealthy Download - Methods for covert app installation:
(1) - Silent app downloads
(2) - Background installation attempts
(3) - Package installer abuse
(4) - ROOT or Accessibility service abuse
Examples: LIBSKIN, TOASTAMIGO patterns
Look for: Download manager usage, package installer calls, hidden installation attempts.""",

    9: """Aggressive Advertising - Methods implementing malicious ad behavior:
(1) - Fake click generation (GhostClicker pattern)
(2) - Forced ad displays
(3) - Background ad loading
(4) - Click fraud implementation
Look for: dispatchTouchEvent abuse, ad library manipulation, screen overlay for ads, click simulation.""",

    10: """Miner - Methods implementing cryptocurrency mining:
(1) - CPU intensive operations
(2) - Cryptocurrency mining code
(3) - Mining pool connections
Examples: HiddenMiner, JSMiner patterns
Look for: High CPU usage patterns, mining library imports, cryptocurrency pool URLs.""",

    11: """Tricky Behavior - Methods implementing evasion techniques:
(1) - Icon/label manipulation
(2) - App hiding mechanisms
(3) - Settings modification
(4) - False uninstall messages
Example: Maikspy error message pattern
Look for: Package visibility changes, settings modifications, fake error messages.""",

    12: """Premium Service Abuse - Methods implementing WAP-Click fraud:
(1) - Automatic premium service subscription
(2) - Hidden browser operations
(3) - WAP-Click abuse
Example: Joker malware pattern
Look for: WAP billing APIs, hidden WebView operations, premium number subscriptions."""
}

# Behavior-specific query templates for better semantic matching
BEHAVIOR_QUERIES = {
    1: """
    Malicious Android class that performs Privacy Stealing by accessing sensitive user data.
    
    Look for:
    - Permission requests: READ_CONTACTS, READ_SMS, ACCESS_FINE_LOCATION, READ_PHONE_STATE
    - API usage: ContentResolver.query() on ContactsContract, TelephonyManager, LocationManager
    - Data extraction: contact names/numbers, SMS messages, GPS coordinates, IMEI/IMSI
    - Network exfiltration: HTTP POST/GET to external servers, hardcoded IP addresses
    - Data serialization: JSON arrays of contacts, SMS, location data
    - Immediate permission abuse: accessing data right after permission grant
    """,
    
    2: """
    Malicious Android class that performs SMS/CALL Abuse by manipulating SMS and phone call functionality.
    
    Look for:
    - SMS operations: SmsManager.sendTextMessage(), content://sms queries, SMS_RECEIVED broadcasts
    - Call manipulation: TelephonyManager, call log access, phone state monitoring
    - Permission requests: SEND_SMS, READ_SMS, CALL_PHONE, READ_CALL_LOG
    - Interception: BroadcastReceiver for SMS/calls, message deletion, call blocking
    - Network communication: sending SMS to premium numbers, call forwarding
    """,
    
    3: """
    Malicious Android class that performs Remote Control by enabling C&C server communication.
    
    Look for:
    - Network connections: Socket connections, HttpURLConnection, OkHttp, WebSocket
    - Remote endpoints: hardcoded IP addresses, suspicious domains, C&C servers
    - Command handling: command parsing, dynamic code execution, remote instructions
    - Background services: Service creation, persistent connections, command polling
    - Dynamic loading: DexClassLoader, reflection, code injection
    """,
    
    4: """
    Malicious Android class that performs Bank/Financial Stealing by implementing banking trojan functionality.
    
    Look for:
    - Overlay attacks: SYSTEM_ALERT_WINDOW permission, transparent activities, phishing windows
    - Banking app detection: package name checks, activity monitoring, app targeting
    - Credential theft: input field monitoring, screen capture, overlay forms
    - Financial data: credit card input, banking credentials, payment information
    - Network exfiltration: sending financial data to external servers
    """,
    
    5: """
    Malicious Android class that performs Ransomware behavior by encrypting files and demanding payment.
    
    Look for:
    - File encryption: encryption algorithms, file system operations, data scrambling
    - Screen locking: device admin, screen lock mechanisms, ransom notes
    - Payment demands: Bitcoin addresses, payment instructions, ransom messages
    - File access: READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE permissions
    - Anti-recovery: file deletion, backup prevention, persistence mechanisms
    """,
    
    6: """
    Malicious Android class that performs Accessibility Abuse by exploiting accessibility services.
    
    Look for:
    - Accessibility service: AccessibilityService implementation, service registration
    - UI automation: performGlobalAction(), window content monitoring, automated clicks
    - Screen interaction: dispatchTouchEvent(), gesture simulation, UI manipulation
    - Silent operations: background installation, permission bypass, automated actions
    - Overlay abuse: SYSTEM_ALERT_WINDOW with accessibility, transparent windows
    """,
    
    7: """
    Malicious Android class that performs Privilege Escalation by attempting to gain elevated privileges.
    
    Look for:
    - Root exploits: su command execution, root checking, privilege escalation
    - System modifications: system file access, /system directory operations
    - Admin rights: device admin requests, system app installation
    - Exploit techniques: Dirty COW, buffer overflows, kernel exploits
    - Persistence: system-level persistence, boot-time execution
    """,
    
    8: """
    Malicious Android class that performs Stealthy Download by covertly installing apps.
    
    Look for:
    - Silent downloads: DownloadManager, background file downloads
    - Package installation: PackageInstaller, INSTALL_PACKAGES permission
    - Hidden installation: background installation, user-unaware app installation
    - Download URLs: hardcoded download links, malicious app sources
    - Installation bypass: ROOT abuse, accessibility service abuse for installation
    """,
    
    9: """
    Malicious Android class that performs Aggressive Advertising by implementing malicious ad behavior.
    
    Look for:
    - Fake clicks: dispatchTouchEvent() abuse, click simulation, ghost clicks
    - Ad manipulation: ad library abuse, forced ad displays, background ad loading
    - Click fraud: automated ad clicks, revenue generation, ad network abuse
    - Overlay ads: screen overlays for ads, invisible ad clicks
    - Ad library integration: malicious ad SDKs, aggressive ad loading
    """,
    
    10: """
    Malicious Android class that performs Cryptocurrency Mining by implementing mining operations.
    
    Look for:
    - CPU intensive operations: high CPU usage, background computation, mining algorithms
    - Mining libraries: cryptocurrency mining code, mining pool connections
    - Network connections: mining pool URLs, cryptocurrency APIs
    - Background services: persistent mining services, battery drain
    - Mining indicators: hash rate monitoring, mining difficulty, cryptocurrency addresses
    """,
    
    11: """
    Malicious Android class that performs Tricky Behavior by implementing evasion techniques.
    
    Look for:
    - App hiding: icon manipulation, label changes, package visibility modifications
    - Settings manipulation: system settings changes, app settings modifications
    - Fake messages: false uninstall messages, fake error dialogs, deceptive UI
    - Evasion techniques: anti-analysis, anti-debugging, detection avoidance
    - Persistence: hidden persistence mechanisms, stealthy operation
    """,
    
    12: """
    Malicious Android class that performs Premium Service Abuse by implementing WAP-Click fraud.
    
    Look for:
    - Premium subscriptions: automatic premium service enrollment, WAP billing
    - Hidden browsers: WebView operations, hidden browser windows, background browsing
    - WAP abuse: premium number subscriptions, automatic billing, hidden charges
    - Network requests: premium service endpoints, billing APIs, subscription calls
    - User deception: hidden operations, unaware billing, premium service abuse
    """
}

# System configuration
CONFIG = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "qwen/qwen3-235b-a22b-2507",
        "embedding_model": "text-embedding-ada-002",
        "temperature": 0.1,
        "max_tokens": 4069,
        "concurrency_limit": 16
    },
    "huggingface": {
        "embedding_model": "NovaSearch/stella_en_1.5B_v5"
    },
    "vectorstore": {
        "type": "chroma",
        "persist_directory": "./smali_vectorstore",
        "collection_name": "smali_classes"
    },
    "retrieval": {
        "top_k_classes": 5,
        "top_k_methods_per_class": 10,
        "relevance_threshold": 0.85
    },
    "processing": {
        "chunk_size": 32000,
        "chunk_overlap": 2000,
        "max_file_size": 1024 * 1024  # 1MB
    },
    "output": {
        "format": "json",
        "include_explanations": True,
        "output_dir": "./output"
    },
    "langfuse": {
        "public_key": os.getenv("LANGFUSE_API_KEY"),
        "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
        "project_id": os.getenv("LANGFUSE_PROJECT_ID"),
        "prompt_names": {
            "smali_class_description_system_prompt": "smali_class_description/system_prompt",
            "smali_class_description_prompt": "smali_class_description/prompt",
            "class_relevance_system_prompt": "class_relevance/system_prompt",
            "class_relevance_prompt": "class_relevance/prompt",
            "method_analysis_system_prompt": "method_analysis/system_prompt",
            "method_analysis_prompt": "method_analysis/prompt"
        }
    }
} 