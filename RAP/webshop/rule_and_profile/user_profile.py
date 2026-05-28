"""
User Profile definition for WebShop environment
Defines user attributes corresponding to the 10 business rules
"""

from typing import List

# Payment method options
PAYMENT_METHODS = ["Visa", "MasterCard", "PayPal", "Prepaid", "Gift Card"]

# Account status options
ACCOUNT_STATUSES = ["active", "restricted"]

# Country options
COUNTRIES = ["allowed", "not_allowed"]


class UserProfile:
    """Represents a user with attributes corresponding to the 10 business rules"""
    
    def __init__(self, profile_id: str, age: int = 25, 
                 country: str = "allowed", is_verified: bool = True,
                 payment_method: str = "Visa", failed_payment_attempts: int = 0,
                 credit_score: int = 700, account_age_days: int = 365,
                 account_status: str = "active", return_rate: float = 0.0,
                 total_purchase_amount: float = 0.0):
        self.profile_id = profile_id
        # Core identity fields
        self.age = age  # User age
        self.country = country  # Country eligibility marker (allowed/not_allowed)
        self.is_verified = is_verified  # Whether user verification is complete
        
        # Payment-related fields
        self.payment_method = payment_method  # Payment method (Visa, MasterCard, PayPal, Prepaid, Gift Card)
        self.failed_payment_attempts = failed_payment_attempts  # Number of failed payment attempts
        
        # Credit and account behavior fields
        self.credit_score = credit_score  # Credit score (0-850)
        self.account_age_days = account_age_days  # Account age in days
        self.account_status = account_status  # Account status (active/restricted)
        self.return_rate = return_rate  # Return rate (0-100%)
        self.total_purchase_amount = total_purchase_amount  # Historical total purchase amount
    
    def to_dict(self):
        return {
            'profile_id': self.profile_id,
            'age': self.age,
            'country': self.country,
            'is_verified': self.is_verified,
            'payment_method': self.payment_method,
            'failed_payment_attempts': self.failed_payment_attempts,
            'credit_score': self.credit_score,
            'account_age_days': self.account_age_days,
            'account_status': self.account_status,
            'return_rate': self.return_rate,
            'total_purchase_amount': self.total_purchase_amount
        }
    
    def __str__(self):
        return (f"Profile({self.profile_id}: age={self.age}, "
                f"country={self.country}, verified={self.is_verified}, "
                f"payment={self.payment_method}, failed_payments={self.failed_payment_attempts}, "
                f"credit_score={self.credit_score}, account_age={self.account_age_days}d, "
                f"status={self.account_status}, return_rate={self.return_rate:.1f}%, "
                f"total_amount=${self.total_purchase_amount:.2f})")

