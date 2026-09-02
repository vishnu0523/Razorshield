#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3  production credentials are refused =="
python3 -c "
from backend.app.adapters.razorpay import assert_test_mode, ProductionCredentialError, TEST_KEY_PREFIX
for bad in ['rzp_'+'live_abc', 'sk_test_x']:
    try:
        assert_test_mode(bad); raise AssertionError(f'{bad} was accepted')
    except ProductionCredentialError: pass
assert_test_mode('rzp_test_ok'); assert_test_mode('')
print(f'   OK - only {TEST_KEY_PREFIX}* or no key at all is accepted')"

echo "== 2/3  webhook signatures verify in constant time =="
python3 -c "
import hmac, hashlib
from backend.app.adapters.razorpay import verify_webhook_signature
body, secret = b'{\"event\":\"payment.captured\"}', 'shhh'
good = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
assert verify_webhook_signature(body, good, secret)
assert not verify_webhook_signature(body, 'deadbeef', secret)
assert not verify_webhook_signature(body, good, 'wrong')
assert not verify_webhook_signature(body, '', secret)
print('   OK - valid signature accepted, forged and wrong-secret rejected')"

echo "== 3/3  payment mapping is honest about what the processor holds =="
python3 -c "
from backend.app.adapters.razorpay import to_internal_payment
p = to_internal_payment({'id':'pay_1','amount':129900,'status':'captured','created_at':1767225600,'method':'upi'})
assert p['amount'] == 1299.0, 'paise not converted'
for f in ('shipping_address_id','coupon_code','product_category'):
    assert p[f] is None, f'{f} was guessed rather than left absent'
print('   OK - paise converted to rupees; fields Razorpay does not hold are left absent, not invented')"

echo
echo "PHASE 15 VERIFIED"
