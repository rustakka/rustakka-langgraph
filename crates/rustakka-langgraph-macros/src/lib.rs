//! Procedural macros for `rustakka-langgraph`.
//!
//! Currently exposes `#[derive(GraphState)]` which produces a `StateSchema`
//! impl mapping struct fields to channels and reducer registrations.
//! `#[node]` is reserved for future ergonomic node registration.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

/// Derive `GraphState` for a struct so it can be used as the typed schema
/// passed to `StateGraph<S>`.
///
/// Each field becomes a channel of the same name. By default the channel
/// uses `LastValue<T>`; annotate fields with `#[reducer = "add_messages"]` or
/// `#[reducer = "topic"]` etc. (resolution happens at runtime via the
/// reducer registry).
#[proc_macro_derive(GraphState, attributes(reducer, channel))]
pub fn derive_graph_state(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let fields = match &ast.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(n) => &n.named,
            _ => return syn_err("GraphState requires a struct with named fields"),
        },
        _ => return syn_err("GraphState can only be derived on structs"),
    };

    let field_idents: Vec<_> = fields.iter().filter_map(|f| f.ident.clone()).collect();
    let field_names: Vec<String> = field_idents.iter().map(|i| i.to_string()).collect();
    let reducers: Vec<String> = fields
        .iter()
        .map(|f| {
            for a in &f.attrs {
                if a.path().is_ident("reducer") {
                    if let Ok(meta) = a.meta.require_name_value() {
                        if let syn::Expr::Lit(lit) = &meta.value {
                            if let syn::Lit::Str(s) = &lit.lit {
                                return s.value();
                            }
                        }
                    }
                }
            }
            "last_value".to_string()
        })
        .collect();

    let expanded = quote! {
        impl ::rustakka_langgraph_core::state::GraphState for #name {
            fn channel_specs() -> ::std::vec::Vec<::rustakka_langgraph_core::state::ChannelSpec> {
                vec![
                    #(
                        ::rustakka_langgraph_core::state::ChannelSpec {
                            name: #field_names.into(),
                            reducer: #reducers.into(),
                        },
                    )*
                ]
            }

            fn to_values(&self) -> ::std::collections::BTreeMap<String, ::serde_json::Value> {
                let mut m = ::std::collections::BTreeMap::new();
                #( m.insert(#field_names.into(), ::serde_json::to_value(&self.#field_idents).unwrap_or(::serde_json::Value::Null)); )*
                m
            }
        }
    };
    expanded.into()
}

fn syn_err(msg: &str) -> TokenStream {
    let m = msg.to_string();
    quote!(compile_error!(#m);).into()
}
