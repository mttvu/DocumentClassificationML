import os

from faker import Faker
from InvoiceGenerator.api import Invoice, Item, Client, Provider, Creator
from InvoiceGenerator.pdf import SimpleInvoice, ProformaInvoice, CorrectingInvoice


fake = Faker('nl_NL')
os.environ["INVOICE_LANG"] = "fr"

client = Client(fake.company(), address=fake.street_address(), zip_code=str(fake.postcode()), city=fake.city())
provider = Provider(fake.company(), bank_account=str(fake.iban()), bank_code=str(fake.random_int(0, 1000)), address=fake.street_address(), zip_code=str(fake.postcode()), city=fake.city())
creator = Creator(fake.name())

invoice = Invoice(client, provider, creator)
invoice.currency = '€'
for j in range(fake.random_digit()):
    invoice.add_item(
        Item(fake.random_digit(), fake.random_int(0, 1000), fake.word(ext_word_list=None), unit='pieces', tax=21))

pdf = SimpleInvoice(invoice)
pdf.gen("invoive.pdf")

