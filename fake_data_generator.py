import random
from faker import Faker
fake = Faker(['de_DE'])


def fake_tax_data():

    dummy_tax_data = []

    # ADDRESS INFORMATION
    dummy_tax_data.extend((' '.join(fake.address().split('\n')), fake.city(), fake.street_name(), fake.street_address(),
                           fake.postcode()))

    # BANK INFORMATION
    dummy_tax_data.extend((fake.bank_country(), fake.iban(), fake.swift8()))

    # BAR CODE INFORMATION
    dummy_tax_data.extend((fake.ean8(), fake.ean8()))

    # COMPANY INFORMATION
    dummy_tax_data.extend((fake.company(), fake.company_suffix(), fake.catch_phrase()))

    # CREDIT CARD INFORMATION
    dummy_tax_data.extend((fake.credit_card_provider(), fake.credit_card_number(), fake.credit_card_expire(),
                           fake.credit_card_security_code()))

    # CURRENCY INFORMATION
    # dummy_tax_data.extend(fake.currency_name())

    # DATE INFORMATION
    dummy_tax_data.extend((str(fake.date_of_birth()), str(fake.month_name()), str(fake.year()), str(fake.date_this_year())))

    # INTERNET INFORMATION
    dummy_tax_data.extend((fake.email(), fake.company_email(), fake.free_email(), fake.user_name(), fake.tld(),
                           fake.domain_name()))

    # ISBN INFORMATION
    dummy_tax_data.extend((fake.isbn10(), fake.isbn13()))

    # JOB TITLE INFORMATION
    dummy_tax_data.extend((fake.job(), fake.job()))

    # NAME INFORMATION
    dummy_tax_data.extend((fake.name(), fake.name_female(), fake.first_name_male(), fake.last_name_male()))

    # PHONE NUMBER INFORMATION
    dummy_tax_data.extend((str(fake.phone_number()), str(fake.country_calling_code())))

    # RANDOM TEXT

    dummy_tax_data.extend((fake.sentence(nb_words=6), fake.sentence(nb_words=6)))

    return dummy_tax_data






